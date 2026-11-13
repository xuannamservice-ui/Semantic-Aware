"""Convergence of the MMSE outage in the number of Zernike terms J and in the aperture grid (primary simulator).

Common random numbers: one draw of standard normals z (length of the largest J) and of the scintillation factors per
slot; the channel with J terms uses the leading J - 3 normals (the Cholesky factor is lower triangular, so the first
coefficients are identical across J). Configurations T8R8, T6R8, T6R11, T8R11 at their link-budget operating points.
Result: results/jconv_<s2>.json with the outage per (grid, J), the paired differences to the largest (grid, J) and
their standard errors.
usage: python run_jconv.py --s2 2.5 --N 2000000 --workers 12 --grids 64,96 --J 435,595,780
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
CFGS = ("T8R8", "T6R8", "T6R11", "T8R11")
_CH = None


def work(args):
    seed, s2, n, batch, grids, Js = args
    global _CH
    if _CH is None:
        _CH = {(g, J): of.OAMChannel(N=g, j_max=J, modes=rm.TX8, rx_modes=rm.RX_ALL) for g in grids for J in Js}
    rng = np.random.default_rng(seed)
    keys = list(_CH)
    jmax = max(Js)
    fail = {k: {c: np.zeros(n, bool) for c in CFGS} for k in keys}
    done = 0
    while done < n:
        b = min(batch, n - done)
        z = rng.standard_normal((b, jmax - of.J_MIN + 1))
        I = _CH[keys[0]].scint(rng, b, s2)
        for k in keys:
            ch = _CH[k]
            G = ch.power_coupling(ch.coeffs(z[:, :ch.j_list.size], s2)) * I[:, None, :]
            for c in CFGS:
                tx, rx = rm.CFG[c]
                Gs = G[:, [rm.RX_ALL.index(m) for m in rx]][:, :, [rm.TX8.index(m) for m in tx]]
                lam, V2 = of.gram_eig(Gs)
                fail[k][c][done:done + b] = (of.sinr_mmse_eig(lam, V2, rm.snr0(len(tx))) < rm.GTH).any(1)
        done += b
    return {f"{g}_{J}": {c: np.packbits(fail[(g, J)][c]) for c in CFGS} for (g, J) in keys}, n


def main(s2, N, workers, batch, grids, Js, seed):
    t0 = time.time()
    per = N // workers
    seeds = [int(s.generate_state(1)[0]) for s in np.random.SeedSequence(seed).spawn(workers)]
    with Pool(workers) as pool:
        res = pool.map(work, [(sd, s2, per, batch, grids, Js) for sd in seeds])
    keys = list(res[0][0])
    ref = f"{max(grids)}_{max(Js)}"
    out = {"s2": s2, "N": per * workers, "grids": grids, "J": Js, "seed": seed, "reference": ref, "configs": {}}
    for c in CFGS:
        f = {k: np.concatenate([np.unpackbits(r[0][k][c])[:per].astype(bool) for r in res]) for k in keys}
        n = f[ref].size
        row = {}
        for k in keys:
            d = f[k].astype(float) - f[ref].astype(float)
            row[k] = {"outage": float(f[k].mean()), "events": int(f[k].sum()),
                      "diff_to_ref": float(d.mean()), "se_diff": float(d.std(ddof=1) / np.sqrt(n))}
        out["configs"][c] = row
        print(c, {k: f"{v['outage']:.3e} (d {v['diff_to_ref']:+.2e} +- {v['se_diff']:.1e})" for k, v in row.items()}, flush=True)
    out["runtime_s"] = time.time() - t0
    with open(os.path.join(OUT, f"jconv_{s2}.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"done in {out['runtime_s']:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--s2", type=float, default=2.5)
    ap.add_argument("--N", type=int, default=2_000_000)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--batch", type=int, default=500)
    ap.add_argument("--grids", default="64,96")
    ap.add_argument("--J", default="435,595,780")
    ap.add_argument("--seed", type=int, default=9901)
    a = ap.parse_args()
    main(a.s2, a.N, a.workers, a.batch, [int(x) for x in a.grids.split(",")], [int(x) for x in a.J.split(",")], a.seed)
