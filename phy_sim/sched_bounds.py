"""Surrogate error terms b(P) and eps_D(P) of the scheduler bound (Theorem S1(iv)) over a set of power allocations.

Theorem S1(iv) needs the failure-probability bias b and the distortion error eps_D of the Block-1 surrogate as suprema
over the power allocations (and hence over the virtual-queue states, through Block 1). This script evaluates them on
fresh Model-A channel samples (sigma_R^2 = 2.5, J = 435, exact MMSE SINR per slot, as run_alloc.py) for
  (i)   the uniform allocation,
  (ii)  every Block-1 closed-form allocation kappa_m = M (chi_m c_m)^q / sum_n (chi_n c_n)^q, q = 1/(1+2d),
        chi_m = V lambda_m + Z theta_m; kappa depends on (V, Z) only through r = Z/V, so a log grid of r in
        [1e-10, 1e10] plus r = 0 and r = inf covers every Z in [0, 1e6] and every V > 0,
  (iii) random allocations on the power simplex (Dirichlet, alpha = 1 ... 100) and near-vertex allocations
        kappa = (1 - e) M e_i + e 1 (the paper imposes only the sum-power constraint (C3), no per-mode limit).
For every allocation the script records the histogram of the failure pattern (bit i = i-th transmitted mode below the
HD-FEC threshold); b, eps_D and their intervals follow from the histograms in the post step.

Exact prefilter (T6R11): the MMSE SINR gamma_m = S_m g_m^T (I + sum_{n != m} S_n g_n g_n^T)^{-1} g_m is linear
increasing in the own power S_m and nonincreasing in every other power S_n (the matrix in brackets grows in the
Loewner order, and inversion is operator-monotone decreasing). For a group of allocations with per-mode bounds
klo_m <= kappa_m <= khi_m, gamma_m(kappa) >= gamma_m(S_m = klo_m^2 S0, S_n = khi_n^2 S0) for every member; if this
lower bound clears the threshold (with a relative margin 1e-6) for all m, no member of the group fails in that slot,
and the slot is counted as failure-free without evaluating the members. Only the remaining slots are evaluated
exactly. The prefilter is checked on the first --nval slots of every worker by evaluating every allocation exactly.

Steps:
  prep (local, needs codec/results):  python sched_bounds.py prep <codec results dir> <phy results dir> <inputs.json>
  run  (server):  python sched_bounds.py run --inputs sched_rev6_inputs.json --cfg T6R11 --N 40000000 --workers 8 --out sb_T6R11.json
  post (local):   python sched_bounds.py post --inputs sched_rev6_inputs.json --runs sb_T6R11.json sb_T8R8.json --out sched_rev6_bounds.json
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

XI = 1e-4 - 2.8e-6          # slack of T6R11 (Section S7): delta_target - abar = 9.7e-5
CFG_TX = {"T6R11": (-3, -2, -1, 1, 2, 3), "T8R8": (-4, -3, -2, -1, 1, 2, 3, 4)}
CFG_RX = {"T6R11": tuple(range(-5, 6)), "T8R8": (-4, -3, -2, -1, 1, 2, 3, 4)}
ANCHOR = {"T6R11": (-1, 2), "T8R8": (-2, -1, 1, 3)}     # best anchor sets of Table S(base_replication), n_A = 2 and 4
KMAX_MEAS = {"T6R11": 2, "T8R8": 4}                       # Pr(Psi<30|K) used for K <= KMAX, K > KMAX counted failed (S8)


def popcount(x):
    x = np.asarray(x, dtype=np.int64)
    c = np.zeros_like(x)
    for i in range(8):
        c += (x >> i) & 1
    return c


# ----------------------------------------------------------------------------------------------------------- prep
def codec_rows(cdir, cfg):
    rows = []
    for f in sorted(glob.glob(f"{cdir}/*/eval_layered_v2a_r1_{cfg}_2.5.json")):
        e = json.load(open(f))
        rows += [dict(scene=e["scene"], view=r["view"], pattern=int(r["pattern"]), K=int(r["K"]), psi=float(r["psnr_vs_full"]))
                 for r in e["rows"] if r["policy"] == "interleaved" and r["method"] == "gen_fp"]
    return rows


def quality_model(rows, cfg):
    """lambda_m (single-stream MSE), q_K, r_K (excess of the measured over the additive distortion), p_q."""
    M = len(CFG_TX[cfg])
    pat = np.array([r["pattern"] for r in rows]); K = np.array([r["K"] for r in rows]); psi = np.array([r["psi"] for r in rows])
    D = 10 ** (-psi / 10)
    lam = np.array([D[pat == (1 << m)].mean() for m in range(M)])
    q = {k: float((psi[K == k] < 30).mean()) for k in range(1, KMAX_MEAS[cfg] + 1)}
    add = np.array([lam[[m for m in range(M) if p >> m & 1]].sum() for p in pat])
    r = {}
    for k in range(2, M + 1):
        s = K == k
        if s.sum() >= 20:
            r[k] = float(D[s].mean() / add[s].mean() - 1)
    last = max(r)
    for k in range(2, M + 1):
        r.setdefault(k, r[last])
    return lam, q, r, float((psi < 30).mean())


def p_fail_patterns(cfg, q):
    tx = CFG_TX[cfg]; M = len(tx)
    A = sum(1 << tx.index(m) for m in ANCHOR[cfg])
    pf = np.zeros(2 ** M)
    for F in range(1, 2 ** M):
        k = bin(F).count("1")
        pf[F] = 1.0 if (F & A) == A or k > KMAX_MEAS[cfg] else q[k]
    return pf, A


def prep(cdir, pdir, out):
    ms = json.load(open(f"{pdir}/modeset_2.5.json"))
    res = {"s2": 2.5, "xi": XI, "configs": {}}
    for cfg in ("T6R11", "T8R8"):
        tx, rx = CFG_TX[cfg], CFG_RX[cfg]; M, Nr = len(tx), len(rx)
        rows = codec_rows(cdir, cfg)
        lam, q, r, pq = quality_model(rows, cfg)
        pf, A = p_fail_patterns(cfg, q)
        rc = ms["configs"][cfg]
        iop = rc["snr_labels"].index("op")
        if cfg == "T8R8":       # run_alloc.py convention
            c = np.array([p["mc_outage"] for p in json.load(open(f"{pdir}/oper_2.5.json"))["per_mode"]])
        else:
            c = np.array(rc["per_mode_count"][iop]) / ms["N"]
        pc = {int(k): v for k, v in rc["op"]["pattern_count"].items()}
        theta = np.zeros(M)
        for m in range(M):
            num = sum(n * pf[F] for F, n in pc.items() if F >> m & 1)
            den = sum(n for F, n in pc.items() if F >> m & 1)
            theta[m] = num / den
        res["configs"][cfg] = dict(tx=tx, rx=rx, M=M, Nr=Nr, d=0.5 * (Nr - M + 1), c_uniform=c.tolist(), lam=lam.tolist(),
                                   theta=theta.tolist(), q=q, r=r, p_q=pq, anchor=ANCHOR[cfg], anchor_mask=A, kmax=KMAX_MEAS[cfg],
                                   p_fail=pf.tolist(), modeset_pattern_count={str(k): v for k, v in pc.items()}, modeset_N=ms["N"],
                                   codec_rows=[[x["scene"], x["view"], x["pattern"], x["K"], x["psi"]] for x in rows])
        print(cfg, "lam", np.round(lam * 1e4, 3), "theta", np.round(theta, 4), "q", q, "r", r, "p_q", round(pq, 4), "c", c)
    json.dump(res, open(out, "w"))


# ----------------------------------------------------------------------------------------------------------- allocations
def closed_form(c, w, d):
    q = 1.0 / (1.0 + 2.0 * d)
    x = (np.asarray(w) * np.asarray(c)) ** q
    return len(c) * x / x.sum()


def allocation_set(inp, seed=5150):
    """Returns list of (kind, label, kappa)."""
    M, d = inp["M"], inp["d"]
    c, lam, th = map(np.asarray, (inp["c_uniform"], inp["lam"], inp["theta"]))
    out = [("uniform", "uniform", np.ones(M))]
    out.append(("closed_form", "r=0", closed_form(c, lam, d)))
    for r in np.logspace(-10, 10, 81):
        out.append(("closed_form", f"r={r:.3g}", closed_form(c, lam + r * th, d)))
    out.append(("closed_form", "r=inf", closed_form(c, th, d)))
    out.append(("closed_form", "equal_weights", closed_form(c, np.ones(M), d)))
    rng = np.random.default_rng(seed)
    for a in (1.0, 3.0, 10.0, 30.0, 100.0):
        for j in range(50):
            out.append(("dirichlet", f"alpha={a:g}#{j}", M * rng.dirichlet(np.full(M, a))))
    for i in range(M):
        for e in (0.95, 0.9, 0.8, 0.5):
            k = e * np.ones(M); k[i] += (1 - e) * M
            out.append(("near_vertex", f"mode{i}_e={e}", k))
    # drop near-duplicate closed forms (large/small r saturate)
    keep, seen = [], []
    for t in out:
        if t[0] == "closed_form" and any(np.max(np.abs(t[2] - s)) < 1e-9 for s in seen):
            continue
        if t[0] in ("closed_form", "uniform"):
            seen.append(t[2])
        keep.append(t)
    return keep


def groups_of(allocs, nsub_edge=0.6):
    """Prefilter groups: closed form + uniform; others by min kappa."""
    edges = [(0.8, 9e9), (0.6, 0.8), (0.4, 0.6), (0.0, 0.4)]
    g = {"cf": [i for i, t in enumerate(allocs) if t[0] in ("uniform", "closed_form")]}
    for lo, hi in edges:
        g[f"min[{lo},{hi})"] = [i for i, t in enumerate(allocs) if t[0] not in ("uniform", "closed_form") and lo <= t[2].min() < hi]
    return {k: v for k, v in g.items() if v}


# ----------------------------------------------------------------------------------------------------------- MC
def mmse_all(GtG, P):
    """GtG (B,M,M), P (A,M) -> SINR (B,A,M), sigma^2 = 1."""
    M = GtG.shape[-1]
    Amat = GtG[:, None, :, :] + (np.eye(M)[None] / P[:, None, :])[None]
    d = np.diagonal(np.linalg.inv(Amat), axis1=-2, axis2=-1)
    return P[None] / d - 1.0


def lower_bound(GtG, klo, khi, S0):
    """Per-mode lower bound of the MMSE SINR over the box klo <= kappa <= khi: (B,M)."""
    M = GtG.shape[-1]
    Pm = np.tile((khi ** 2) * S0, (M, 1))
    Pm[np.arange(M), np.arange(M)] = np.maximum(klo ** 2 * S0, 1e-300)
    g = mmse_all(GtG, Pm)                       # (B, M_vectors, M_modes)
    return g[:, np.arange(M), np.arange(M)]


_W = {}


def worker(args):
    cfg, inp, seed, n, batch, nval, nsub, prefilter = args
    import oamfso as of
    import run_modeset as rm
    if "ch" not in _W:
        _W["ch"] = of.OAMChannel(modes=rm.TX8, rx_modes=rm.RX_ALL)
    ch = _W["ch"]
    tx, rx = inp["tx"], inp["rx"]; M = len(tx)
    ri = [rm.RX_ALL.index(m) for m in rx]; ti = [rm.TX8.index(m) for m in tx]
    S0 = rm.snr0(M); gth = of.gamma_th_from_ber()
    allocs = allocation_set(inp)
    K = np.array([t[2] for t in allocs])
    P = K ** 2 * S0
    grp = groups_of(allocs)
    gb = {g: (K[idx].min(0), K[idx].max(0)) for g, idx in grp.items()}
    nmax = {g: (n if K[idx].min() >= 0.6 else min(n, nsub)) for g, idx in grp.items()}
    hist = np.zeros((len(allocs), 2 ** M), np.int64)
    neval = np.zeros(len(allocs), np.int64)
    surv = {g: 0 for g in grp}; seen = {g: 0 for g in grp}
    val = {"slots": 0, "pruned_slot_failures": 0, "pruned_slots": 0}
    w = 1 << np.arange(M)
    rng = np.random.default_rng(seed)
    done = 0
    while done < n:
        b = min(batch, n - done)
        G = ch.draw(rng, b, inp["s2"])[:, ri][:, :, ti]
        GtG = np.einsum("brm,brn->bmn", G, G)
        for g, idx in grp.items():
            nb = min(b, nmax[g] - done)
            if nb <= 0:
                continue
            Gg = GtG[:nb]
            if prefilter:
                lb = lower_bound(Gg, *gb[g], S0)
                bad = (lb < gth * (1 + 1e-6)).any(1)
            else:
                bad = np.ones(nb, bool)
            seen[g] += nb; surv[g] += int(bad.sum())
            neval[idx] += nb
            sidx = np.flatnonzero(bad)
            for s0 in range(0, sidx.size, 256):
                ss = sidx[s0:s0 + 256]
                for a0 in range(0, len(idx), 64):
                    ai = np.array(idx[a0:a0 + 64])
                    f = mmse_all(Gg[ss], P[ai]) < gth          # (s, a, M)
                    code = (f * w).sum(-1)
                    for j, a in enumerate(ai):
                        hist[a] += np.bincount(code[:, j], minlength=2 ** M)
            hist[idx, 0] += nb - sidx.size                         # pruned slots: no failure for any member
            if prefilter and done < nval:                          # validation of the prefilter
                pr = np.flatnonzero(~bad)
                val["slots"] += nb; val["pruned_slots"] += int(pr.size)
                for s0 in range(0, pr.size, 256):
                    ss = pr[s0:s0 + 256]
                    for a0 in range(0, len(idx), 64):
                        ai = np.array(idx[a0:a0 + 64])
                        val["pruned_slot_failures"] += int((mmse_all(Gg[ss], P[ai]) < gth).any(-1).any(-1).sum())
        done += b
    return hist, neval, surv, seen, val


def run(a):
    from multiprocessing import Pool
    t0 = time.time()
    inp = json.load(open(a.inputs))["configs"][a.cfg]
    inp["s2"] = 2.5
    allocs = allocation_set(inp)
    per = a.N // a.workers
    jobs = [(a.cfg, inp, a.seed + 1000 * i, per, a.batch, a.nval, a.nsub // a.workers, not a.no_prefilter) for i in range(a.workers)]
    with Pool(a.workers) as p:
        outs = p.map(worker, jobs)
    hist = sum(o[0] for o in outs); neval = sum(o[1] for o in outs)
    surv = {g: sum(o[2][g] for o in outs) for g in outs[0][2]}
    seen = {g: sum(o[3][g] for o in outs) for g in outs[0][3]}
    val = {k: sum(o[4][k] for o in outs) for k in outs[0][4]}
    res = dict(cfg=a.cfg, N=per * a.workers, workers=a.workers, seed=a.seed, prefilter=not a.no_prefilter,
               groups={g: dict(slots=seen[g], survivors=surv[g], survivor_fraction=surv[g] / max(seen[g], 1)) for g in seen},
               validation=val, runtime_s=time.time() - t0,
               allocations=[dict(kind=t[0], label=t[1], kappa=t[2].tolist(), n=int(neval[i]),
                                 hist={str(F): int(h) for F, h in enumerate(hist[i]) if h}) for i, t in enumerate(allocs)])
    json.dump(res, open(a.out, "w"))
    print(f"{a.cfg}: N={res['N']} groups={res['groups']} validation={val} runtime={res['runtime_s']:.0f}s", flush=True)


# ----------------------------------------------------------------------------------------------------------- post
def cp(k, n, conf=0.95):
    from scipy.stats import beta
    al = 1 - conf
    lo = 0.0 if k == 0 else beta.ppf(al / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - al / 2, k + 1, n - k)
    return float(lo), float(hi)


def boot_quality(inp, cfg, B, rng):
    """Cluster bootstrap over (scene, view) of lambda, q_K, r_K, p_q."""
    rows = inp["codec_rows"]
    keys = sorted({(r[0], r[1]) for r in rows})
    byk = {}
    for r in rows:
        byk.setdefault((r[0], r[1]), []).append(dict(scene=r[0], view=r[1], pattern=r[2], K=r[3], psi=r[4]))
    out = []
    for _ in range(B):
        pick = rng.integers(0, len(keys), len(keys))
        rr = [x for i in pick for x in byk[keys[i]]]
        try:
            out.append(quality_model(rr, cfg))
        except Exception:
            continue
    return out


def metrics(inp, cfg, hist, n, lam, q, r, pq, kappa):
    """b_paper, b_surr, eps_D(additive), eps_D(model) and outage numbers for one allocation."""
    M = inp["M"]; d = inp["d"]
    F = np.array(sorted(hist), dtype=np.int64); h = np.array([hist[f] for f in sorted(hist)], dtype=float)
    pf = p_fail_patterns(cfg, q)[0][F]
    Kf = popcount(F)
    lsum = np.array([lam[[m for m in range(M) if f >> m & 1]].sum() for f in F])
    rk = np.array([r.get(int(k), 0.0) if k >= 2 else 0.0 for k in Kf])
    pout = h[Kf > 0].sum() / n
    pm = np.array([h[(F >> m) & 1 == 1].sum() / n for m in range(M)])
    pfail = (h * pf).sum() / n
    pmodel = np.asarray(inp["c_uniform"]) * kappa ** (-2 * d)      # per-mode model (eq. poutmodel), calibrated at uniform
    th = np.asarray(inp["theta"])
    return dict(P_out=pout, sum_P_m=pm.sum(), per_mode=pm.tolist(), P_fail_true=pfail,
                b_paper=(pm.sum() - pout) * pq,
                b_surr=abs((th * pmodel).sum() - pfail), P_fail_surr=float((th * pmodel).sum()),
                b_unionq=abs(pm.sum() * pq - pfail),
                eps_D=abs((h * rk * lsum).sum() / n),
                eps_D_model=abs((lam * pmodel).sum() - (h * (1 + rk) * lsum).sum() / n))


def post(a):
    rng = np.random.default_rng(7)
    I = json.load(open(a.inputs))
    out = {"xi": XI, "xi_half": XI / 2, "configs": {}}
    for f in a.runs:
        R = json.load(open(f)); cfg = R["cfg"]; inp = I["configs"][cfg]; M = inp["M"]
        lam0, q0, r0, pq0 = np.asarray(inp["lam"]), {int(k): v for k, v in inp["q"].items()}, {int(k): v for k, v in inp["r"].items()}, inp["p_q"]
        qb = boot_quality(inp, cfg, a.B, rng)
        rows = []
        for al in R["allocations"]:
            hist = {int(k): v for k, v in al["hist"].items()}; n = al["n"]; kap = np.asarray(al["kappa"])
            m0 = metrics(inp, cfg, hist, n, lam0, q0, r0, pq0, kap)
            # CP intervals: P_out, and b_paper = pq * sum_{j>=1} Pr(K >= j+1) (Bonferroni over the M-1 terms)
            F = np.array(list(hist), dtype=np.int64); h = np.array(list(hist.values()), dtype=float); Kf = popcount(F)
            kout = int(h[Kf > 0].sum())
            bl, bh = 0.0, 0.0
            for j in range(1, M):
                kk = int(h[Kf >= j + 1].sum())
                lo, hi = cp(kk, n, 1 - 0.05 / (M - 1))
                bl += lo; bh += hi
            # parametric bootstrap: Poisson counts x cluster-bootstrap quality model
            bs = {k: [] for k in ("b_surr", "eps_D", "eps_D_model", "P_fail_true", "b_unionq")}
            for i in range(min(a.B, len(qb))):
                hb = {Fk: int(c) for Fk, c in zip(F, rng.poisson(h)) if c}
                lam, q, r, pq = qb[i]
                mb = metrics(inp, cfg, hb if hb else {0: n}, n, lam, q, r, pq, kap)
                for k in bs:
                    bs[k].append(mb[k])
            row = dict(kind=al["kind"], label=al["label"], kappa=al["kappa"], n=n, outage_slots=kout, **m0,
                       P_out_cp95=cp(kout, n), b_paper_cp95=[bl * pq0, bh * pq0])
            for k, v in bs.items():
                row[k + "_ci95"] = [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
            rows.append(row)
        summ = {}
        for kind in ("uniform", "closed_form", "dirichlet", "near_vertex", "all"):
            sel = [x for x in rows if kind == "all" or x["kind"] == kind]
            for met in ("b_paper", "b_surr", "eps_D", "eps_D_model"):
                best = max(sel, key=lambda x: x[met])
                ci = best[met + "_cp95"] if met == "b_paper" else best[met + "_ci95"]
                summ[f"{kind}/{met}"] = dict(max=best[met], ci95=ci, argmax=best["label"], kappa=best["kappa"], n=best["n"],
                                              P_out=best["P_out"], below_xi_half=bool(ci[1] < XI / 2))
        cfr = [x for x in rows if x["kind"] in ("closed_form", "uniform")]
        summ["closed_form_kappa_range"] = [float(np.min([x["kappa"] for x in cfr])), float(np.max([x["kappa"] for x in cfr]))]
        out["configs"][cfg] = dict(N=R["N"], groups=R["groups"], validation=R["validation"], runtime_s=R["runtime_s"],
                                   inputs={k: inp[k] for k in ("lam", "theta", "q", "r", "p_q", "c_uniform", "d", "anchor")},
                                   summary=summ, allocations=rows)
        for k, v in summ.items():
            if isinstance(v, dict):
                print(cfg, k, f"{v['max']:.3g}", [f"{x:.3g}" for x in v["ci95"]], v["argmax"], f"Pout={v['P_out']:.3g}")
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "prep":
        prep(*sys.argv[2:5]); sys.exit()
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("run", "post"))
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--cfg", default="T6R11")
    ap.add_argument("--N", type=int, default=40_000_000)
    ap.add_argument("--nsub", type=int, default=4_000_000, help="slots for allocation groups with min kappa < 0.6")
    ap.add_argument("--nval", type=int, default=20_000, help="slots per worker on which the prefilter is validated")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--batch", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=606001)
    ap.add_argument("--no-prefilter", action="store_true")
    ap.add_argument("--runs", nargs="*")
    ap.add_argument("--B", type=int, default=400)
    ap.add_argument("--out", default="sb.json")
    a = ap.parse_args()
    run(a) if a.step == "run" else post(a)
