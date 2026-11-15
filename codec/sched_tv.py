"""Block 0 of the scheduler under time-varying turbulence: wavelength reuse (T8R11W4, 8 streams on 4 carriers) versus
separate carriers (T6R11, 6 streams on 6 carriers).

Inputs (measured): failed-slot rate per configuration and Rytov variance,
    abar_c(s2) = pi_A,c(s2) + sum_K P_c(K; s2) f_c(K),
with P_c(K; s2) and the anchor-pair loss pi_A,c from the link simulator (phy_sim/results/reuse_curve.json for
s2 <= 0.4, beat_<s2>.json / modeset / tolerance for larger s2) and f_c(K) = Pr(Psi < 30 dB | K failed streams) from the
trace-driven codec evaluation (11 scenes, interleaved mapping, concealment network); K beyond the evaluated range
counts as failed (upper bound).
Turbulence: ln s2(t) = ln(m) + ln(A) sin(2 pi (t - 6 h)/24 h) + x(t), x an Ornstein-Uhlenbeck process (std sd,
correlation time tau); Block 0 runs every second on an estimate s2_hat = s2 exp(e), e ~ N(0, est^2).
Policies: static separate, static reuse, threshold (reuse if s2_hat <= th), and the drift-plus-penalty rule
c(t) = argmax_c V SE_c - Z(t) n abar_c(s2_hat) with the virtual queue Z <- max(0, Z + F - n delta), F the failed slots
of the second (Poisson), n = 1000 slots/s, SE_c = streams per carrier.
usage: python sched_tv.py <phy_results_dir> <codec_results_dir> <out.json>
"""
import glob
import json
import math
import sys

import numpy as np
from scipy.signal import lfilter

PHY, COD, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
N_S = 1000                      # slots per second (T_f = 1 ms)
DELTA = 1e-4
DAYS = 10
SE = {"reuse": 2.0, "sep": 1.0}  # streams per optical carrier
RATE = {"reuse": 8 * 9.35, "sep": 6 * 9.35}   # Gb/s


def cond_fail(cfg, src):
    rows = []
    for f in glob.glob(f"{COD}/*/eval_layered_v2a_r1_{cfg}_{src}.json"):
        e = json.load(open(f))
        rows += [r for r in e["rows"] if r["policy"] == "interleaved" and r["method"] == "gen_fp"]
    f = {}
    for K in range(1, 9):
        x = [r["psnr_vs_full"] < 30 for r in rows if r["K"] == K]
        f[K] = float(np.mean(x)) if len(x) >= 30 else 1.0
    return f


fK = {"reuse": cond_fail("T8R11W4", "0.2w"), "sep": cond_fail("T6R11", "2.5")}


def abar(nfail, anchor, f):
    h = np.asarray(nfail, float); h = h / h.sum()
    return anchor + sum(h[K] * f[K] for K in range(1, len(h)))


# failed-slot rate versus s2 for both configurations
pts = {"reuse": {}, "sep": {}}
rc = json.load(open(f"{PHY}/reuse_curve.json"))
for s2, d in rc["curves"].items():
    pts["reuse"][float(s2)] = abar(d["T8R11W4"]["nfail"], d["T8R11W4"]["best_anchor_loss"], fK["reuse"])
    pts["sep"][float(s2)] = abar(d["T6R11"]["nfail"], d["T6R11"]["best_anchor_loss"], fK["sep"])
for s2 in (0.5, 1.0, 2.5):
    b = json.load(open(f"{PHY}/beat_{s2}.json"))["configs"]["T8R11"]["W4"]
    pts["reuse"][s2] = abar(b["nfail"], 0.0, fK["reuse"]) + 0.0     # anchor loss not recorded: lower bound
tol = json.load(open(f"{PHY}/tolerance_2.5.json"))
g = [x for x in tol["configs"]["T6R11"]["grid"] if x["extra_loss_dB"] == 0.0 and x["rin_dB_Hz"] is None][0]
pts["sep"][2.5] = abar(g["nfail_hist"], g["best_anchor_loss"], fK["sep"])
ms1 = json.load(open(f"{PHY}/modeset_1.0.json"))["configs"]["T6R11"]
i = ms1["snr_labels"].index("op")
pts["sep"][1.0] = max(ms1["outage_count"][i] / 1e7, 1e-9)          # K = 1 at most: abar <= outage
FLOOR = 1e-9


def curve(c):
    x = np.array(sorted(pts[c])); y = np.array([max(pts[c][v], FLOOR) for v in x])
    lx, ly = np.log(x), np.log(y)
    return lambda s2: np.exp(np.interp(np.log(s2), lx, ly))    # flat beyond the ends (conservative at low s2)


AB = {c: curve(c) for c in pts}


def profile(rng, m=0.35, A=6.0, sd=0.35, tau=900.0):
    t = np.arange(DAYS * 86400)
    det = math.log(m) + math.log(A) * np.sin(2 * np.pi * (t / 86400.0 - 0.25))
    a = math.exp(-1.0 / tau)
    w = rng.standard_normal(t.size) * sd * math.sqrt(1 - a * a); w[0] = rng.standard_normal() * sd
    x = lfilter([1.0], [1.0, -a], w)
    return np.exp(det + x)


def run(policy, s2, est, rng, V=None, th=None, dq=DELTA):
    s2h = s2 * np.exp(est * rng.standard_normal(s2.size))
    ab_true = {c: AB[c](s2) for c in AB}
    use = np.zeros(s2.size, bool); F = np.zeros(s2.size); Z = 0.0; Zs = np.zeros(s2.size)
    abh = {c: AB[c](s2h) for c in AB}
    lam_r = ab_true["reuse"] * N_S; lam_s = ab_true["sep"] * N_S
    pr = rng.poisson(lam_r); ps = rng.poisson(lam_s)
    for k in range(s2.size):
        if policy == "sep":
            u = False
        elif policy == "reuse":
            u = True
        elif policy == "threshold":
            u = s2h[k] <= th
        else:
            u = V * SE["reuse"] - Z * N_S * abh["reuse"][k] > V * SE["sep"] - Z * N_S * abh["sep"][k]
        use[k] = u; F[k] = pr[k] if u else ps[k]
        Z = max(0.0, Z + F[k] - N_S * dq); Zs[k] = Z
    hours = F.reshape(-1, 3600).sum(1) / (3600 * N_S)
    return {"policy": policy, "V": V, "th": th, "delta_q": dq, "reuse_share": float(use.mean()),
            "se": float(np.where(use, SE["reuse"], SE["sep"]).mean()),
            "rate_gbps": float(np.where(use, RATE["reuse"], RATE["sep"]).mean()),
            "abar": float(F.sum() / (s2.size * N_S)), "hours_above_target": float((hours > DELTA).mean()),
            "worst_hour": float(hours.max()), "Z_mean": float(Zs.mean()), "Z_max": float(Zs.max())}


def main():
    rng = np.random.default_rng(2611)
    s2 = profile(rng)
    res = {"inputs": {"fK": fK, "points": {c: {str(k): v for k, v in sorted(p.items())} for c, p in pts.items()}},
           "profile": {"days": DAYS, "median": float(np.median(s2)), "share_le_0.2": float((s2 <= 0.2).mean()),
                       "share_le_0.3": float((s2 <= 0.3).mean()), "min": float(s2.min()), "max": float(s2.max())},
           "runs": []}
    print(res["profile"]); print({c: {k: f"{v:.2e}" for k, v in sorted(p.items())} for c, p in pts.items()})
    for est in (0.1, 0.3):
        cases = [("sep", {}), ("reuse", {})] + [("threshold", {"th": t}) for t in (0.1, 0.15, 0.2, 0.25, 0.3)] + \
                [("lyapunov", {"V": v}) for v in (1.0, 100.0, 1e4)] + [("lyapunov", {"V": 1.0, "dq": d}) for d in (3e-5, 1e-5)]
        for pol, kw in cases:
            r = run(pol, s2, est, np.random.default_rng(7), **kw); r["est"] = est
            res["runs"].append(r)
            print(est, pol, kw, {k: (round(v, 4) if isinstance(v, float) and v > 1e-3 else v) for k, v in r.items()
                                 if k in ("reuse_share", "se", "rate_gbps", "abar", "hours_above_target", "worst_hour", "Z_max")})
    json.dump(res, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
