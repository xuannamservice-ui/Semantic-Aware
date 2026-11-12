"""Closed-form outage-minimizing power allocation (Corollary 1(b) of the paper) evaluated by Monte Carlo.

With the asymptotic per-mode model P_out,m = c_m (kappa_m^2 gamma0)^(-d), minimizing sum_m P_out,m subject to
sum_m kappa_m = M gives kappa_m = M c_m^q / sum_n c_n^q with q = 1/(1+2d), and lowers the model union bound by the
SNR-independent factor M^(2d) sum c / (sum c^q)^(1+2d). The coefficients c_m are taken from the uniform-power
per-mode Monte Carlo outage at the operating point (oper_<s2>.json for the square sorter, modeset_<s2>.json
otherwise). The allocations are compared on common channel realizations with uniform power and, for the square
sorter, with the allocation obtained numerically from the tail-matched surrogate (power_<s2>.json).
Result: sim/results/alloc_<cfg>_<s2>.json
"""
import argparse
import json
import math
import os
import time

import numpy as np

import oamfso as of
from run_modeset import CFG, snr0

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
GTH = of.gamma_th_from_ber()


def load(name):
    with open(os.path.join(OUT, name + ".json")) as f:
        return json.load(f)


def main(cfg, s2, N, batch, seed):
    t0 = time.time()
    tx, rx = CFG[cfg]
    M, Nr = len(tx), len(rx)
    d = 0.5 * (Nr - M + 1)
    S0 = snr0(M)
    if cfg == "T8R8":
        pm = np.array([p["mc_outage"] for p in load(f"oper_{s2}")["per_mode"]])
    else:
        r = load(f"modeset_{s2}")["configs"][cfg]
        iop = r["snr_labels"].index("op")
        pm = np.array(r["per_mode_count"][iop]) / load(f"modeset_{s2}")["N"]
    q = 1.0 / (1.0 + 2.0 * d)
    k_cf = M * pm ** q / np.sum(pm ** q)
    gain_model = M ** (2 * d) * pm.sum() / np.sum(pm ** q) ** (1 + 2 * d)
    allocs = {"uniform": np.ones(M), "closed_form": k_cf}
    if cfg == "T8R8":
        allocs["surrogate_numerical"] = np.array(load(f"power_{s2}")["allocations"]["outage_min"]["kappa"])
    ch = of.OAMChannel(modes=tx, rx_modes=rx)
    rng = np.random.default_rng(seed)
    acc = {a: {"out": 0, "pm": np.zeros(M, np.int64), "se": 0.0} for a in allocs}
    ind = {a: np.zeros(N, bool) for a in allocs}
    done = 0
    while done < N:
        b = min(batch, N - done)
        G = ch.draw(rng, b, s2)
        for a, k in allocs.items():
            g = of.sinr_mmse(G, k ** 2 * S0)
            fail = g < GTH
            acc[a]["out"] += int(fail.any(1).sum())
            acc[a]["pm"] += fail.sum(0)
            acc[a]["se"] += float(np.log2(1 + g).sum())
            ind[a][done:done + b] = fail.any(1)
        done += b
    res = {"cfg": cfg, "s2": s2, "N": N, "d": d, "q": q, "snr0_dB": 10 * math.log10(S0), "c_source_per_mode_outage": pm.tolist(),
           "union_bound_gain_model": gain_model, "allocations": {}}
    base = ind["uniform"].astype(float)
    for a, k in allocs.items():
        o = acc[a]["out"] / N
        diff = ind[a].astype(float) - base
        res["allocations"][a] = {"kappa": k.tolist(), "P_mW": (100.0 / M * k).tolist(), "sys_outage": o, "count": acc[a]["out"],
                                 "per_mode_outage": (acc[a]["pm"] / N).tolist(), "union_bound_mc": float(acc[a]["pm"].sum() / N),
                                 "mean_se": acc[a]["se"] / N,
                                 "paired_diff_vs_uniform": {"mean": float(diff.mean()), "ci95": 1.96 * float(diff.std(ddof=1)) / math.sqrt(N)}}
        print(f"{cfg} s2={s2} {a}: P_mW {np.round(100.0 / M * k, 2)} outage {o:.5g} sum per-mode {acc[a]['pm'].sum() / N:.5g} "
              f"mean SE {acc[a]['se'] / N:.2f} diff {diff.mean():+.3g}", flush=True)
    res["runtime_s"] = time.time() - t0
    with open(os.path.join(OUT, f"alloc_{cfg}_{s2}.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(f"{cfg} s2={s2}: model union-bound gain {gain_model:.4f}; done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="T8R8")
    ap.add_argument("--s2", type=float, required=True)
    ap.add_argument("--N", type=int, default=2_000_000)
    ap.add_argument("--batch", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=4404)
    a = ap.parse_args()
    main(a.cfg, a.s2, a.N, a.batch, a.seed)
