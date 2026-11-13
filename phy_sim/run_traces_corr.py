"""Temporally correlated loss traces: run lengths of outage slots for a given turbulence coherence time.

The per-slot statistics of run_modeset.py treat slots as independent channel realizations. Here consecutive
slots of duration T_f are linked by a first-order Gauss-Markov model with correlation rho = exp(-T_f/tau_c):
the normalized Zernike vector z, and the Gaussian copulas of the scintillation factors X and Y_n, each follow
u_{t+1} = rho u_t + sqrt(1 - rho^2) w_t. The marginal distribution of every slot is exactly that of
run_modeset.py, so the outage probability per slot is unchanged; only the temporal structure differs.
Chains of n_slots slots are simulated in parallel. --rho-model kolmogorov uses rho = exp(-(T_f/tau_c)^(5/3)).
Result: sim/results/traces_corr_<s2>_<tau>ms<tag>.json and the per-slot failure patterns (npz, uint8, bit i = i-th transmitted mode).
"""
import argparse
import json
import math
import os
import time

import numpy as np
from scipy import special, stats

import oamfso as of
from run_modeset import CFG, RX_ALL, TX8, snr0

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
GTH = of.gamma_th_from_ber()
CFGS = ("T8R8", "T6R8", "T4R8", "T8R11", "T6R11")
ANCHOR_PAIRS = ((-1, 1), (-2, 2))


def run_lengths(x):
    """Lengths of the runs of True in each row of a boolean array (chains x slots)."""
    out = []
    for row in x:
        if not row.any():
            continue
        d = np.diff(np.concatenate(([0], row.astype(np.int8), [0])))
        st, en = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
        out.append(en - st)
    return np.concatenate(out) if out else np.zeros(0, int)


def main(s2, tau_ms, n_chains, n_slots, tf_ms, seed, rho_model="exp", tag=""):
    t0 = time.time()
    rho = math.exp(-tf_ms / tau_ms) if rho_model == "exp" else math.exp(-(tf_ms / tau_ms) ** (5 / 3))
    sq = math.sqrt(1 - rho * rho)
    base = of.OAMChannel()
    gr = base.grid
    U = {m: of.lg_mode(m, gr["r"], gr["phi"], base.w) for m in RX_ALL}
    Q = (np.conj(np.stack([U[m] for m in RX_ALL]))[:, None, :] * np.stack([U[m] for m in TX8])[None, :, :] * gr["dA"]
         ).reshape(len(RX_ALL) * len(TX8), -1).T
    Qr, Qi = np.ascontiguousarray(Q.real), np.ascontiguousarray(Q.imag)
    al, be = base.gg_params(s2)
    rng = np.random.default_rng(seed)
    J = base.j_list.size
    z = rng.standard_normal((n_chains, J))
    ux = rng.standard_normal(n_chains)
    uy = rng.standard_normal((n_chains, len(TX8)))
    fails = {k: np.zeros((n_chains, n_slots, len(CFG[k][0])), bool) for k in CFGS}
    for t in range(n_slots):
        if t:
            z = rho * z + sq * rng.standard_normal(z.shape)
            ux = rho * ux + sq * rng.standard_normal(ux.shape)
            uy = rho * uy + sq * rng.standard_normal(uy.shape)
        X = stats.gamma.ppf(special.ndtr(ux), al, scale=1 / al)
        Y = stats.gamma.ppf(special.ndtr(uy), be, scale=1 / be)
        Phi = base.coeffs(z, s2) @ base.Z
        c, s = np.cos(Phi), np.sin(Phi)
        Gall = ((c @ Qr - s @ Qi) ** 2 + (c @ Qi + s @ Qr) ** 2).reshape(n_chains, len(RX_ALL), len(TX8)) * (X[:, None] * Y)[:, None, :]
        for k in CFGS:
            tx, rx = CFG[k]
            G = Gall[:, [RX_ALL.index(m) for m in rx]][:, :, [TX8.index(m) for m in tx]]
            g = of.sinr_mmse(G, snr0(len(tx)))
            fails[k][:, t] = g < GTH
    res = {"s2": s2, "tau_c_ms": tau_ms, "T_f_ms": tf_ms, "rho": rho, "rho_model": rho_model, "seed": seed, "n_chains": n_chains, "n_slots": n_slots, "configs": {}}
    for k in CFGS:
        f = fails[k]
        out = f.any(2)
        rl = run_lengths(out)
        row = {"outage": float(out.mean()), "n_runs": int(rl.size), "mean_run_slots": float(rl.mean()) if rl.size else None,
               "p_run_ge": {str(L): float((rl >= L).mean()) if rl.size else None for L in (2, 5, 10, 20)},
               "max_run_slots": int(rl.max()) if rl.size else 0,
               "p_next_slot_outage_given_outage": float((out[:, 1:] & out[:, :-1]).sum() / max(out[:, :-1].sum(), 1))}
        tx = CFG[k][0]
        for pr in ANCHOR_PAIRS:
            if set(pr) <= set(tx):
                lost = f[:, :, tx.index(pr[0])] & f[:, :, tx.index(pr[1])]
                rla = run_lengths(lost)
                row[f"anchor_{pr[0]}_{pr[1]}"] = {"loss": float(lost.mean()), "mean_run_slots": float(rla.mean()) if rla.size else None,
                                                   "n_runs": int(rla.size)}
        a0, a1 = out[:, :-1].astype(float), out[:, 1:].astype(float)
        row["outage_slots"] = int(out.sum())
        row["adjacent_outage_pairs"] = int((out[:, 1:] & out[:, :-1]).sum())
        row["rho_out_lag1"] = float(np.corrcoef(a0.ravel(), a1.ravel())[0, 1]) if out.any() else None
        res["configs"][k] = row
        print(f"s2={s2} tau={tau_ms} ms {k}: outage {row['outage']:.4g}, mean run {row['mean_run_slots']}, "
              f"P(next|out) {row['p_next_slot_outage_given_outage']:.3f}", flush=True)
    res["runtime_s"] = time.time() - t0
    # per-slot failure pattern (bit i = i-th transmitted mode of the configuration failed), uint8 (M <= 8)
    w = {k: (1 << np.arange(fails[k].shape[2], dtype=np.uint16)) for k in CFGS}
    np.savez_compressed(os.path.join(OUT, f"traces_corr_{s2}_{tau_ms:g}ms{tag}.npz"),
                        **{k: (fails[k].astype(np.uint16) @ w[k]).astype(np.uint8) for k in CFGS},
                        **{f"{k}_tx": np.array(CFG[k][0]) for k in CFGS}, n_slots=n_slots)
    with open(os.path.join(OUT, f"traces_corr_{s2}_{tau_ms:g}ms{tag}.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--s2", type=float, required=True)
    ap.add_argument("--tau", type=float, required=True, help="coherence time (ms)")
    ap.add_argument("--chains", type=int, default=2000)
    ap.add_argument("--slots", type=int, default=250)
    ap.add_argument("--tf", type=float, default=1.0, help="slot duration (ms)")
    ap.add_argument("--seed", type=int, default=6601)
    ap.add_argument("--rho-model", default="exp", choices=("exp", "kolmogorov"))
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    main(a.s2, a.tau, a.chains, a.slots, a.tf, a.seed, a.rho_model, a.tag)
