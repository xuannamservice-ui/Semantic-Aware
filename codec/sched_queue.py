"""Hard-mode virtual queue of the scheduler on Model-B correlated losses (T6R11, sigma_R^2 = 2.5, J = 435).

The channel follows Model B of run_traces_corr.py (Gauss-Markov update of the Zernike vector and of the Gaussian
copulas of X and Y_n, Kolmogorov temporal correlation rho = exp(-(T_f/tau_c)^(5/3)), T_f = 1 ms), simulated slot by
slot for independent chains. In every slot each policy transmits with its own power allocation kappa(t), the exact
MMSE SINR gives its failure pattern F(t), and the failed-slot indicator is
    a(t) = 1 if the anchor pair (-1, +2) is lost, or K = |F| >= 3, or U(t) < Pr(Psi < 30 | K) for K = 1, 2,
with Pr(Psi < 30 | K) = 1.5% and 7.8% measured on the T6R11 traces (Section S8) and U(t) uniform, common to all
policies (paired comparison). Variant a_cJ uses min(1, 1.27 Pr(Psi < 30 | K)) for K = 1, 2, the Zernike extrapolation
factor that Section S8 applies to the quality term (it does not scale the K >= 3 and anchor terms).
Policies:
  uniform      kappa = 1 (static, Block 0 only)
  cf_static    Block-1 closed form at Z = 0, kappa = M (lambda_m c_m)^q / sum (lambda_n c_n)^q (soft-mode allocation)
  dyn_V<V>     Block-1 closed form with chi_m(t) = V lambda_m + Z(t) theta_m, reacting to the virtual queue
The virtual queue of every policy follows Algorithm S1: Z(t+1) = max(0, Z(t) + a_hat(t) - delta), delta = 1e-4,
with a_hat(t) = 1 for an anchor loss or K >= 3 and Pr(Psi < 30 | K) otherwise; the queue driven by the realized a(t)
is recorded as well. lambda_m, theta_m, c_m and Pr(Psi < 30 | K) are read from sched_rev6_inputs.json
(phy_sim/sched_bounds.py prep).
Steps:
  run  (server): python sched_queue.py run --inputs sched_rev6_inputs.json --tau 2.5 --jobs 16 --workers 16 --chains 16 --slots 400000 --out sq_2.5
  post:          python sched_queue.py post --runs sq_2.5 sq_10 --out sched_rev6_queue.json
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import glob
import json
import math
import sys
import time

import numpy as np

DELTA = 1e-4
CJ = 1.2721
TX = (-3, -2, -1, 1, 2, 3)
RX = tuple(range(-5, 6))
VS = (0.01, 1.0, 100.0)
WINDOWS = (1000, 10000, 100000)


def closed_form(c, w, d):
    q = 1.0 / (1.0 + 2.0 * d)
    x = (w * c) ** q
    return len(c) * x / x.sum(-1, keepdims=True)


def worker(args):
    wid, inp, tau, n_chains, n_slots, seed, phy_dir = args
    sys.path.insert(0, phy_dir)
    import oamfso as of
    import run_modeset as rm
    from scipy import special
    s2 = 2.5
    rho = math.exp(-(1.0 / tau) ** (5 / 3)); sq = math.sqrt(1 - rho * rho)
    base = of.OAMChannel()
    gr = base.grid
    U = {m: of.lg_mode(m, gr["r"], gr["phi"], base.w) for m in set(RX) | set(TX)}
    Q = (np.conj(np.stack([U[m] for m in RX]))[:, None, :] * np.stack([U[m] for m in TX])[None, :, :] * gr["dA"]).reshape(len(RX) * len(TX), -1).T
    Qr, Qi = np.ascontiguousarray(Q.real), np.ascontiguousarray(Q.imag)
    al, be = base.gg_params(s2)
    M = len(TX); S0 = rm.snr0(M); gth = of.gamma_th_from_ber()
    c = np.asarray(inp["c_uniform"]); lam = np.asarray(inp["lam"]); th = np.asarray(inp["theta"]); d = inp["d"]
    A = inp["anchor_mask"]; kmax = inp["kmax"]
    qk = np.zeros(M + 1); qk[1:kmax + 1] = [inp["q"][str(k)] for k in range(1, kmax + 1)]; qk[kmax + 1:] = 1.0
    qcj = np.minimum(1.0, CJ * qk); qcj[kmax + 1:] = 1.0
    pols = ["uniform", "cf_static"] + [f"dyn_V{V:g}" for V in VS]
    k_static = {"uniform": np.ones(M), "cf_static": closed_form(c, lam, d)}
    rng = np.random.default_rng(seed)
    J = base.j_list.size
    z = rng.standard_normal((n_chains, J)); ux = rng.standard_normal(n_chains); uy = rng.standard_normal((n_chains, M))
    Zh = {p: np.zeros(n_chains) for p in pols}; Zr = {p: np.zeros(n_chains) for p in pols}
    st = {p: dict(zpos=0, zpos_real=0, zmax=0.0, zmax_real=0.0, zsum=0.0, kappa_dev_max=0.0) for p in pols}
    ev = {p: [] for p in pols}
    w = 1 << np.arange(M)
    eye = np.eye(M)
    t0 = time.time()
    for t in range(n_slots):
        if t:
            z = rho * z + sq * rng.standard_normal(z.shape)
            ux = rho * ux + sq * rng.standard_normal(ux.shape)
            uy = rho * uy + sq * rng.standard_normal(uy.shape)
        X = special.gammaincinv(al, special.ndtr(ux)) / al
        Y = special.gammaincinv(be, special.ndtr(uy)) / be
        Phi = base.coeffs(z, s2) @ base.Z
        cs, sn = np.cos(Phi), np.sin(Phi)
        G = ((cs @ Qr - sn @ Qi) ** 2 + (cs @ Qi + sn @ Qr) ** 2).reshape(n_chains, len(RX), M) * (X[:, None] * Y)[:, None, :]
        GtG = np.einsum("brm,brn->bmn", G, G)
        u = rng.random(n_chains)
        for p in pols:
            if p in k_static:
                kap = np.broadcast_to(k_static[p], (n_chains, M))
            else:
                V = float(p[5:])
                kap = closed_form(c, V * lam + Zh[p][:, None] * th, d)
                dev = np.abs(kap - k_static["cf_static"]).max()
                if dev > st[p]["kappa_dev_max"]:
                    st[p]["kappa_dev_max"] = float(dev)
            P = kap ** 2 * S0
            dg = np.diagonal(np.linalg.inv(GtG + eye / P[:, None, :]), axis1=1, axis2=2)
            code = ((P / dg - 1.0 < gth) * w).sum(1)
            zh, zr = Zh[p], Zr[p]
            st[p]["zpos"] += int((zh > 0).sum()); st[p]["zpos_real"] += int((zr > 0).sum()); st[p]["zsum"] += float(zh.sum())
            if code.any():
                K = np.zeros(n_chains, np.int64)
                for i in range(M):
                    K += (code >> i) & 1
                hard = (code != 0) & (((code & A) == A) | (K > kmax))
                ahat = np.where(hard, 1.0, qk[K] * (code != 0))
                a = (code != 0) & (hard | (u < qk[K]))
                acj = (code != 0) & (hard | (u < qcj[K]))
                nz = np.flatnonzero(code)
                for b in nz:
                    ev[p].append((t, wid * n_chains + b, int(code[b]), int(a[b]), int(acj[b]), float(ahat[b])))
            else:
                ahat = np.zeros(n_chains); a = np.zeros(n_chains, bool)
            Zh[p] = np.maximum(0.0, zh + ahat - DELTA)
            Zr[p] = np.maximum(0.0, zr + a - DELTA)
            st[p]["zmax"] = max(st[p]["zmax"], float(Zh[p].max())); st[p]["zmax_real"] = max(st[p]["zmax_real"], float(Zr[p].max()))
        if wid == 0 and t % 50000 == 0:
            print(f"tau={tau} worker0 slot {t}/{n_slots} {time.time() - t0:.0f}s", flush=True)
    return dict(wid=wid, stats=st, events={p: np.array(ev[p], dtype=float).reshape(-1, 6) for p in pols}, runtime_s=time.time() - t0)


def run(a):
    from multiprocessing import Pool
    t0 = time.time()
    inp = json.load(open(a.inputs))["configs"]["T6R11"]
    jobs = [(i, inp, a.tau, a.chains, a.slots, a.seed + 7919 * i + int(10 * a.tau), a.phy) for i in range(a.jobs)]
    with Pool(a.workers) as p:
        outs = p.map(worker, jobs)
    pols = list(outs[0]["events"])
    np.savez_compressed(a.out + ".npz", **{p: np.concatenate([o["events"][p] for o in outs]) for p in pols})
    stats = {p: dict(zpos=sum(o["stats"][p]["zpos"] for o in outs), zpos_real=sum(o["stats"][p]["zpos_real"] for o in outs),
                     zsum=sum(o["stats"][p]["zsum"] for o in outs), zmax=max(o["stats"][p]["zmax"] for o in outs),
                     zmax_real=max(o["stats"][p]["zmax_real"] for o in outs), kappa_dev_max=max(o["stats"][p]["kappa_dev_max"] for o in outs))
             for p in pols}
    json.dump(dict(tau_c_ms=a.tau, rho=math.exp(-(1.0 / a.tau) ** (5 / 3)), n_chains=a.jobs * a.chains, chain_slots=a.slots,
                   total_slots=a.jobs * a.chains * a.slots, seed=a.seed, delta=DELTA, cJ=CJ, V=VS,
                   kappa_cf_static=closed_form(np.asarray(inp["c_uniform"]), np.asarray(inp["lam"]), inp["d"]).tolist(),
                   stats=stats, runtime_s=time.time() - t0), open(a.out + ".json", "w"), indent=1)
    print(f"done {a.out} in {time.time() - t0:.0f}s", flush=True)


# ------------------------------------------------------------------------------------------------ post
def window_stats(times_by_chain, n_chains, L, W):
    nw = L // W
    cnt = np.zeros(n_chains * nw, np.int64)
    smax = 0
    for ch, tt in times_by_chain.items():
        tt = np.sort(tt)
        tt_c = tt[tt < nw * W]
        np.add.at(cnt, ch * nw + (tt_c // W).astype(np.int64), 1)
        if tt.size:
            j = np.searchsorted(tt, tt + W, side="left")
            smax = max(smax, int((j - np.arange(tt.size)).max()))
    thr = int(math.floor(DELTA * W)) + 1          # count exceeding delta * W
    r = dict(W=W, n_windows=int(cnt.size), max_count=int(cnt.max()), sliding_max_count=smax,
             frac_ge1=float((cnt >= 1).mean()), frac_ge2=float((cnt >= 2).mean()), frac_ge11=float((cnt >= 11).mean()),
             exceed_threshold=thr, frac_exceed=float((cnt >= thr).mean()))
    for k in ("frac_ge1", "frac_ge2", "frac_ge11", "frac_exceed"):
        key = {"frac_ge1": 1, "frac_ge2": 2, "frac_ge11": 11, "frac_exceed": thr}[k]
        n_ex = int((cnt >= key).sum())
        r[k + "_cp95"] = cp(n_ex, cnt.size)
    return r


def cp(k, n, conf=0.95):
    from scipy.stats import beta
    al = 1 - conf
    return [0.0 if k == 0 else float(beta.ppf(al / 2, k, n - k + 1)), 1.0 if k == n else float(beta.ppf(1 - al / 2, k + 1, n - k))]


def longest_burst(times_by_chain):
    best = 0
    for tt in times_by_chain.values():
        tt = np.sort(tt)
        if tt.size == 0:
            continue
        brk = np.flatnonzero(np.diff(tt) != 1)
        runs = np.diff(np.concatenate(([-1], brk, [tt.size - 1])))
        best = max(best, int(runs.max()))
    return best


def post(a):
    out = {"delta": DELTA, "runs": {}}
    for r in a.runs:
        meta = json.load(open(r + ".json")); E = np.load(r + ".npz")
        n_ch, L, N = meta["n_chains"], meta["chain_slots"], meta["total_slots"]
        res = dict(tau_c_ms=meta["tau_c_ms"], n_chains=n_ch, chain_slots=L, total_slots=N, policies={})
        for p in E.files:
            e = E[p]
            t, ch, code, av, acj = e[:, 0].astype(np.int64), e[:, 1].astype(np.int64), e[:, 2].astype(np.int64), e[:, 3] > 0, e[:, 4] > 0
            pr = dict(outage_slots=int(t.size), P_out=t.size / N)
            # P(next | out)
            key = ch * L + t
            ks = set(key.tolist())
            pr["P_next_given_out"] = float(np.mean([(k + 1) in ks for k in key if (k % L) < L - 1])) if t.size else None
            K = np.zeros_like(code)
            for i in range(6):
                K += (code >> i) & 1
            pr["K_hist"] = {int(k): int((K == k).sum()) for k in np.unique(K)}
            for name, msk in (("a", av), ("a_cJ", acj)):
                tbc = {}
                for c_, t_ in zip(ch[msk], t[msk]):
                    tbc.setdefault(int(c_), []).append(int(t_))
                tbc = {k: np.array(v) for k, v in tbc.items()}
                per_chain = np.zeros(n_ch)
                for k, v in tbc.items():
                    per_chain[k] = v.size / L
                nf = int(msk.sum())
                se = per_chain.std(ddof=1) / math.sqrt(n_ch)
                d = dict(failures=nf, abar=nf / N, abar_ci95_chains=[nf / N - 1.96 * se, nf / N + 1.96 * se],
                         abar_cp95_iid=cp(nf, N), per_chain_abar_max=float(per_chain.max()),
                         frac_chains_above_delta=float((per_chain > DELTA).mean()), longest_burst=longest_burst(tbc),
                         windows=[window_stats(tbc, n_ch, L, W) for W in WINDOWS if W <= L])
                pr[name] = d
            s = meta["stats"][p]
            pr["queue"] = dict(frac_Z_pos=s["zpos"] / N, frac_Z_pos_cp95=cp(s["zpos"], N), mean_Z=s["zsum"] / N, max_Z=s["zmax"],
                               frac_Z_pos_realized_a=s["zpos_real"] / N, max_Z_realized_a=s["zmax_real"],
                               kappa_dev_max_vs_cf_static=s["kappa_dev_max"])
            pr["_per_chain_fail"] = np.bincount(ch[av], minlength=n_ch)
            res["policies"][p] = pr
            print(f"tau={meta['tau_c_ms']} {p}: Pout={pr['P_out']:.3g} Pnext={pr['P_next_given_out']} abar={pr['a']['abar']:.3g} "
                  f"{[round(x, 8) for x in pr['a']['abar_ci95_chains']]} Z>0 {pr['queue']['frac_Z_pos']:.3g} maxZ {s['zmax']:.3g} "
                  f"burst {pr['a']['longest_burst']} " + " ".join(f"W={x['W']}: ex={x['frac_exceed']:.3g} max={x['max_count']} smax={x['sliding_max_count']}" for x in pr["a"]["windows"]),
                  flush=True)
        # paired differences of the failure counts (common channel and common U), t-interval over the chains
        res["paired"] = {}
        for p, ref in [(p, r_) for r_ in ("uniform", "cf_static") for p in res["policies"] if p != r_]:
            dlt = (res["policies"][p]["_per_chain_fail"] - res["policies"][ref]["_per_chain_fail"]) / L
            se = dlt.std(ddof=1) / math.sqrt(n_ch)
            res["paired"][f"{p}-{ref}"] = dict(mean=float(dlt.mean()), ci95=[float(dlt.mean() - 1.96 * se), float(dlt.mean() + 1.96 * se)],
                                               chains_differing=int((dlt != 0).sum()))
        for p in res["policies"]:
            res["policies"][p].pop("_per_chain_fail")
        print(json.dumps(res["paired"]), flush=True)
        out["runs"][str(meta["tau_c_ms"])] = res
        json.dump(out, open(a.out, "w"), indent=1)
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("run", "post"))
    ap.add_argument("--inputs", default="sched_rev6_inputs.json")
    ap.add_argument("--tau", type=float, default=2.5)
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--chains", type=int, default=16)
    ap.add_argument("--slots", type=int, default=400_000)
    ap.add_argument("--seed", type=int, default=880001)
    ap.add_argument("--phy", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "phy_sim"))
    ap.add_argument("--runs", nargs="*")
    ap.add_argument("--out", default="sq")
    a = ap.parse_args()
    run(a) if a.step == "run" else post(a)
