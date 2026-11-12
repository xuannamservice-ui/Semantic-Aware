"""Hardware tolerance grid: MMSE outage versus laser RIN and extra sorter insertion loss (primary simulator).

Same channel and receiver as run_modeset.py (common random numbers across all grid points and configurations).
For each configuration and each (extra loss dL, RIN) the operating SNR is run_modeset.snr0(M, dL, RIN); the stage
records system outage, the number of failed streams, and joint failures of every mode pair (anchor loss).
Result: results/tolerance_<s2>.json
usage: python run_tolerance.py --s2 2.5 --N 2000000 --workers 10
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(v, "1")
import argparse
import json
import time
from multiprocessing import Pool

import numpy as np

import oamfso as of
import run_modeset as rm

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
CFGS = ("T8R8", "T8R11", "T6R8", "T6R11", "T4R8")
DL_DB = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
RIN = (None, -170.0, -165.0, -160.0, -155.0, -150.0)
_CH = None


def grid(M):
    return [(dl, r, rm.snr0(M, dl, r)) for dl in DL_DB for r in RIN]


def work(args):
    seed, s2, n, batch = args
    global _CH
    if _CH is None:
        _CH = of.OAMChannel(modes=rm.TX8, rx_modes=rm.RX_ALL)
    ch = _CH
    rng = np.random.default_rng(seed)
    acc = {}
    for k in CFGS:
        M = len(rm.CFG[k][0]); ng = len(DL_DB) * len(RIN)
        acc[k] = {"out": np.zeros(ng, np.int64), "nfail": np.zeros((ng, M + 1), np.int64),
                  "pair": np.zeros((ng, M, M), np.int64)}
    done = 0
    while done < n:
        b = min(batch, n - done)
        z = rng.standard_normal((b, ch.j_list.size))
        I = ch.scint(rng, b, s2)
        G = ch.power_coupling(ch.coeffs(z, s2)) * I[:, None, :]
        for k in CFGS:
            tx, rx = rm.CFG[k]
            Gs = G[:, [rm.RX_ALL.index(m) for m in rx]][:, :, [rm.TX8.index(m) for m in tx]]
            lam, V2 = of.gram_eig(Gs)
            for i, (_, _, snr) in enumerate(grid(len(tx))):
                fail = of.sinr_mmse_eig(lam, V2, snr) < rm.GTH
                a = acc[k]
                a["out"][i] += int(fail.any(1).sum())
                a["nfail"][i] += np.bincount(fail.sum(1), minlength=len(tx) + 1)
                fi = fail.astype(np.int64)
                a["pair"][i] += fi.T @ fi
        done += b
    return acc


def main(s2, N, workers, batch, seed):
    t0 = time.time()
    per = N // workers
    seeds = [int(s.generate_state(1)[0]) for s in np.random.SeedSequence(seed).spawn(workers)]
    with Pool(workers) as pool:
        res = pool.map(work, [(sd, s2, per, batch) for sd in seeds])
    n_tot = per * workers
    out = {"s2": s2, "N": n_tot, "seed": seed, "gamma_th_dB": 10 * np.log10(rm.GTH), "configs": {}}
    for k in CFGS:
        tx, rx = rm.CFG[k]
        tot = {f: sum(r[k][f] for r in res) for f in ("out", "nfail", "pair")}
        rows = []
        for i, (dl, r, snr) in enumerate(grid(len(tx))):
            pair = tot["pair"][i] / n_tot
            off = [(tx[a], tx[b], float(pair[a, b])) for a in range(len(tx)) for b in range(a + 1, len(tx))]
            best = min(off, key=lambda t: t[2])
            rows.append({"extra_loss_dB": dl, "rin_dB_Hz": r, "snr0_dB": float(10 * np.log10(snr)),
                         "outage": float(tot["out"][i] / n_tot), "outage_events": int(tot["out"][i]),
                         "nfail_hist": tot["nfail"][i].tolist(),
                         "best_anchor_pair": [best[0], best[1]], "best_anchor_loss": best[2],
                         "pair_joint_fail": pair.tolist()})
        out["configs"][k] = {"tx": list(tx), "rx": list(rx), "grid": rows}
        nom = rows[0]
        print(f"{k}: nominal outage {nom['outage']:.3e} ({nom['outage_events']} events), "
              f"best pair {nom['best_anchor_pair']} loss {nom['best_anchor_loss']:.2e}", flush=True)
    out["runtime_s"] = time.time() - t0
    with open(os.path.join(OUT, f"tolerance_{s2}.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"done in {out['runtime_s']:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--s2", type=float, default=2.5)
    ap.add_argument("--N", type=int, default=2_000_000)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--batch", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=8801)
    a = ap.parse_args()
    main(a.s2, a.N, a.workers, a.batch, a.seed)
