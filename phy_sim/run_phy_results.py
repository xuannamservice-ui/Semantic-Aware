"""Regenerate every physical-layer result of the manuscript from the model in oamfso.py.

Stages (run all, or a subset with --stages; Monte Carlo stages can be restricted to some Rytov
variances with --s2 so that independent processes can run in parallel):
  analytics : Gamma-Gamma parameters, validity and accuracy of the high-SNR EC bound, Fig. 1
  jsweep    : Zernike truncation order on common random numbers, Fig. 9
  sens      : sensitivity of the operating-point outage to the modelling assumptions
  power     : power allocation (uniform, outage-minimizing, capacity-maximizing)
  fig2      : system outage vs reference SNR (Monte Carlo and surrogate union bounds), Fig. 2
  oper      : operating point (link-budget SNR): SINR/outage table, lower-tail validation, KS/KL,
              mean crosstalk matrix (Fig. 3) and quantile plots (Fig. 13)
  plot      : figures from the stored results
Results go to sim/results/<stage>_<s2>.json, figures to the manuscript folder.
"""
import argparse
import json
import math
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate, optimize, special

import oamfso as of

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
FIGDIR = os.path.dirname(HERE)
os.makedirs(OUT, exist_ok=True)

S2_MAIN = (0.2, 1.0, 2.5)
S2_ALL = (0.2, 0.5, 1.0, 1.5, 2.5)
COLORS = {0.2: "#2a78d6", 1.0: "#eb6834", 2.5: "#1baf7a"}
NAMES = {0.2: "Weak", 1.0: "Moderate", 2.5: "Strong"}
LB = of.link_budget()
SNR0 = LB["snr0"]
GTH = of.gamma_th_from_ber()

plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.edgecolor": "#888", "axes.linewidth": 0.6})


def save(name, obj):
    with open(os.path.join(OUT, name + ".json"), "w") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else float(o))


def load(name):
    with open(os.path.join(OUT, name + ".json")) as f:
        return json.load(f)


def have(name):
    return os.path.exists(os.path.join(OUT, name + ".json"))


def style(ax):
    ax.grid(color="#e5e5e5", lw=0.5); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def draw_many(ch, rng, N, s2, batch=2000, dtype=np.float32):
    G = np.empty((N, ch.M, ch.M), dtype=dtype)
    for i in range(0, N, batch):
        b = min(batch, N - i)
        G[i:i + b] = ch.draw(rng, b, s2)
    return G


def mmse_chunks(G, P, chunk=100000):
    out = np.empty(G.shape[:2], dtype=np.float64)
    for i in range(0, G.shape[0], chunk):
        out[i:i + chunk] = of.sinr_mmse(G[i:i + chunk].astype(np.float64), P)
    return out


def sinr_mmse_sd(G, kappa, chunk=100000):
    """MMSE SINR with per-port shot noise at the instantaneous mean port power sum_n G_mn kappa_n P_rx
    (signal-dependent noise); signal powers in units of the nominal noise variance."""
    th, sh = LB["thermal_var_A2"], LB["shot_var_A2"]
    out = np.empty(G.shape[:2])
    for i in range(0, G.shape[0], chunk):
        Gc = G[i:i + chunk].astype(np.float64)
        v = (th + sh * np.einsum("bmn,n->bm", Gc, kappa)) / (th + sh)
        out[i:i + chunk] = of.sinr_mmse(Gc / np.sqrt(v)[:, :, None], kappa ** 2 * SNR0)
    return out


def binom_ci(cnt, N):
    p = cnt / N
    return 1.96 * math.sqrt(p * (1 - p) / N)


# =============================================================================== analytics
def ec_exact_per_mode(p, gbar, a, b):
    """-ln E[(1+gbar I^2)^-p]/(p ln 2) for unit-mean Gamma-Gamma I (numerical quadrature)."""
    f = lambda I: (1 + gbar * I * I) ** (-p) * of.gg_pdf_I(I, a, b)
    val = 0.0
    edges = [0, 1e-3, 1e-2, 0.1, 0.5, 1, 2, 5, 20, np.inf]
    for lo, hi in zip(edges[:-1], edges[1:]):
        val += integrate.quad(f, lo, hi, limit=200)[0]
    return -math.log(val) / (p * math.log(2))


def ergodic_per_mode(gbar, a, b):
    f = lambda I: math.log2(1 + gbar * I * I) * of.gg_pdf_I(I, a, b)
    edges = [0, 1e-3, 1e-2, 0.1, 0.5, 1, 2, 5, 20, np.inf]
    return sum(integrate.quad(f, lo, hi, limit=200)[0] for lo, hi in zip(edges[:-1], edges[1:]))


def ec_hs_per_mode(p, gbar, a, b):
    if 2 * p >= min(a, b):
        return None
    lnM = (2 * p * math.log(a * b) + special.gammaln(a - 2 * p) + special.gammaln(b - 2 * p)
           - special.gammaln(a) - special.gammaln(b) - p * math.log(gbar))
    return -lnM / (p * math.log(2))


def stage_analytics():
    res = {"link_budget": LB, "gamma_th": GTH, "gamma_th_dB": 10 * math.log10(GTH), "J_MAX": of.J_MAX}
    w0, zR, w = of.beam_params()
    res["beam"] = {"w0_cm": 100 * w0, "w_cm": 100 * w, "D_scint_cm": 200 * w,
                   "d2": of.K_WAVE * (2 * w) ** 2 / (4 * of.L_LINK), "d2_aperture": of.K_WAVE * of.D_R ** 2 / (4 * of.L_LINK)}
    C = of.noll_cov(4, 17)
    res["noll"] = {"c_n": {n: of.noll_cn(n) for n in range(1, 8)},
                   "Delta1": of.noll_residual(1), "Delta3": of.noll_residual(3), f"Delta_{of.J_MAX}": of.noll_residual(of.J_MAX),
                   "tilt_share": 2 * of.noll_cn(1) / of.noll_residual(1),
                   "corr_4_11": float(C[0, 7] / math.sqrt(C[0, 0] * C[7, 7])),
                   "corr_7_17": float(C[3, 13] / math.sqrt(C[3, 3] * C[13, 13]))}
    rows = []
    for s2 in S2_ALL:
        ap, bp = of.gg_point(s2); aa, ba = of.gg_aperture(s2); tb = of.turbulence(s2)
        pmax = min(aa, ba) / 2
        rows.append({"s2": s2, "Cn2": tb["Cn2"], "r0_cm": 100 * tb["r0"], "D_over_r0": tb["D_over_r0"],
                     "residual_phase_var_rad2": of.noll_residual(of.J_MAX) * tb["D_over_r0"] ** (5 / 3),
                     "alpha_point": ap, "beta_point": bp, "SI_point": of.scint_index(ap, bp),
                     "alpha": aa, "beta": ba, "SI": of.scint_index(aa, ba),
                     "p_max": pmax, "s_max": pmax * math.log(2), "theta_max": pmax * math.log(2) / 1e7})
    res["gg_table"] = rows
    acc = []
    for gbar_dB in (20.0, round(LB["snr0_dB"], 1)):
        gbar = 10 ** (gbar_dB / 10)
        for s2 in S2_MAIN:
            a, b = of.gg_aperture(s2)
            for p in (0.5, 1.44, 3.0, 5.0):
                ex = ec_exact_per_mode(p, gbar, a, b); hs = ec_hs_per_mode(p, gbar, a, b)
                acc.append({"gbar_dB": gbar_dB, "s2": s2, "p": p, "exact": ex, "hs": hs,
                            "rel_err": None if hs is None else (hs - ex) / ex})
    res["ec_accuracy"] = acc
    snr_dB = np.arange(0, 72.5, 2.5)
    fig1 = {"snr_dB": snr_dB.tolist(), "curves": {}}
    for s2 in S2_MAIN:
        a, b = of.gg_aperture(s2)
        for key in ("shannon", 0.1, 1.0, 10.0):
            ys = []
            for d in snr_dB:
                gbar = 10 ** (d / 10)
                c = ergodic_per_mode(gbar, a, b) if key == "shannon" else ec_exact_per_mode(float(key) / math.log(2), gbar, a, b)
                ys.append(c * 8 * 10e9 / 1e9)
            fig1["curves"][f"{s2}_{key}"] = ys
    res["fig1"] = fig1
    loss = {}
    for d in (20.0, LB["snr0_dB"]):
        gbar = 10 ** (d / 10)
        for s2 in S2_MAIN:
            a, b = of.gg_aperture(s2); sh = ergodic_per_mode(gbar, a, b)
            loss[f"{d:.1f}_{s2}"] = {str(s): 1 - ec_exact_per_mode(s / math.log(2), gbar, a, b) / sh for s in (0.1, 1.0, 10.0)}
    res["ec_loss_vs_shannon"] = loss
    save("analytics", res)
    fig, axs = plt.subplots(1, 3, figsize=(7.16, 2.4), sharey=True, layout="constrained")
    sty = [("shannon", "Ergodic (Shannon)", "#555555", "--", None), (0.1, r"$s_\theta=0.1$", "#6da7ec", "-", "o"),
           (1.0, r"$s_\theta=1$", "#2a78d6", "-", "s"), (10.0, r"$s_\theta=10$", "#184f95", "-", "^")]
    for ax, s2 in zip(axs, S2_MAIN):
        for k, lab, c, ls, mk in sty:
            ax.plot(snr_dB, fig1["curves"][f"{s2}_{k}"], ls=ls, color=c, lw=1.5, marker=mk, ms=3.5, markevery=4, label=lab)
        ax.axvline(LB["snr0_dB"], color="#999999", lw=0.8, ls=":")
        ax.set_title(f"{NAMES[s2]} ($\\sigma_R^2={s2}$)", fontsize=9); style(ax)
    axs[0].set_ylabel(r"Total $E_C$ (Gbit/s)"); axs[0].legend(frameon=False, fontsize=7, loc="upper left")
    fig.supxlabel(r"Average electrical SNR per mode, $\bar\gamma$ (dB)", fontsize=9)
    fig.savefig(os.path.join(FIGDIR, "fig1_effective_capacity.png"), dpi=300)
    return res


# =============================================================================== J sweep
J_SWEEP = (6, 10, 15, 21, 28, 45, 66, 91, 120, 153, 231, 325, 435)


def stage_jsweep(s2_list, N=100_000):
    """Outage and mean coupling versus the truncation order J on common random numbers: one normal
    vector z for the largest J per sample, lower J use its leading columns (nested Cholesky factor),
    and the scintillation draw is shared by all J."""
    J_ref = J_SWEEP[-1]
    ch = of.OAMChannel(j_max=J_ref)
    for s2 in s2_list:
        t0 = time.time()
        rng = np.random.default_rng({0.2: 1123, 1.0: 1234, 2.5: 2345}[s2])
        ind = np.zeros((len(J_SWEEP), N), dtype=bool)
        pc = np.zeros((len(J_SWEEP), ch.M, ch.M))
        for i in range(0, N, 2000):
            b = min(2000, N - i)
            z = rng.standard_normal((b, ch.j_list.size))
            I = ch.scint(rng, b, s2)
            for k, J in enumerate(J_SWEEP):
                Pc = ch.power_coupling(ch.coeffs(z, s2, J - of.J_MIN + 1))
                g = of.sinr_mmse(Pc * I[:, None, :], SNR0)
                ind[k, i:i + b] = (g < GTH).any(1)
                pc[k] += Pc.sum(0)
        rows = []
        k_mod = J_SWEEP.index(of.J_MAX)
        for k, J in enumerate(J_SWEEP):
            d = ind[k].astype(float) - ind[-1].astype(float)
            dm = ind[k].astype(float) - ind[k_mod].astype(float)
            diag = np.diag(pc[k] / N)
            rows.append({"J": J, "outage": float(ind[k].mean()), "count": int(ind[k].sum()), "ci95": binom_ci(int(ind[k].sum()), N),
                         "residual_var": of.noll_residual(J),
                         "diff_vs_ref": float(d.mean()), "diff_vs_ref_ci95": 1.96 * float(d.std(ddof=1)) / math.sqrt(N),
                         "diff_vs_model": float(dm.mean()), "diff_vs_model_ci95": 1.96 * float(dm.std(ddof=1)) / math.sqrt(N),
                         "Pc_diag_by_absm": [float(np.mean([diag[i] for i, m in enumerate(of.MODES) if abs(m) == q])) for q in (1, 2, 3, 4)]})
            print(f"jsweep s2={s2} J={J}: outage {ind[k].mean():.4g}, diff vs J={J_ref} {d.mean():+.4g} +- {rows[-1]['diff_vs_ref_ci95']:.2g}, "
                  f"Pc diag {np.round(rows[-1]['Pc_diag_by_absm'], 3)}", flush=True)
        save(f"jsweep_{s2}", {"N": N, "s2": s2, "J_ref": J_ref, "J_model": of.J_MAX, "rows": rows})
        print(f"jsweep s2={s2} done in {time.time()-t0:.0f}s", flush=True)


def plot_jsweep():
    fig, axs = plt.subplots(2, 1, figsize=(3.5, 4.3), sharex=True)
    for s2 in S2_MAIN:
        if not have(f"jsweep_{s2}"):
            continue
        rows = load(f"jsweep_{s2}")["rows"]; c = COLORS[s2]
        x = [r["J"] for r in rows]
        axs[0].errorbar(x, [r["outage"] for r in rows], yerr=[r["ci95"] for r in rows], color=c, marker="o", ms=3.5,
                        lw=1.4, capsize=2, label=f"$\\sigma_R^2={s2}$")
        for q, ls in ((1, "-"), (4, "--")):
            axs[1].plot(x, [r["Pc_diag_by_absm"][q - 1] for r in rows], color=c, ls=ls, marker="o", ms=3, lw=1.3,
                        label=f"$\\sigma_R^2={s2}$, $|m|={q}$")
    ticks = [6, 10, 28, 66, 153, 435]
    for ax in axs:
        ax.axvline(of.J_MAX, color="#999999", lw=0.8, ls=":"); style(ax)
        ax.set_xscale("log"); ax.set_xticks(ticks); ax.set_xticklabels([str(t) for t in ticks]); ax.minorticks_off()
    axs[1].set_xlabel("Zernike truncation order $J$")
    axs[0].set_ylabel("System outage"); axs[0].legend(frameon=False, fontsize=8, loc="upper left")
    axs[1].set_ylabel(r"$\mathbb{E}[|h_{m,m}|^2]$"); axs[1].legend(frameon=False, fontsize=7.5, loc="upper right", ncol=1)
    fig.tight_layout(); fig.savefig(os.path.join(FIGDIR, "fig9_zernike_order.png"), dpi=300)


# =============================================================================== sensitivity
def coupling_displaced(ch, z, s2, d):
    """Power coupling with a lateral displacement d (per sample, metres) of the received beam relative
    to the mode-sorter axis; the phase screen stays in receiver coordinates."""
    gr = ch.grid
    UrefH = np.ascontiguousarray((np.conj(np.stack([of.lg_mode(m, gr["r"], gr["phi"], ch.w) for m in ch.modes])) * gr["dA"]).T)
    E = np.exp(1j * (ch.coeffs(z, s2) @ ch.Z))
    x = gr["x"][None, :] - d[:, :1]; y = gr["y"][None, :] - d[:, 1:]
    r_, ph = np.hypot(x, y), np.arctan2(y, x)
    X = np.stack([of.lg_mode(m, r_, ph, ch.w) for m in ch.modes], axis=1) * E[:, None, :]     # (b, n, k)
    h = np.swapaxes(X @ UrefH, 1, 2)                                                         # (b, m, n)
    return h.real ** 2 + h.imag ** 2


def stage_sens(s2_list, N=200_000, seed=505, batch=1000):
    """Operating-point MMSE outage under alternative modelling assumptions, on common random numbers:
    the phase normals z, the displacement normals and, where the scintillation model is unchanged,
    the scintillation draw are shared by all cases."""
    th, sh = LB["thermal_var_A2"], LB["shot_var_A2"]
    rin_db = -155.0
    snr_rin = LB["photocurrent_A"] ** 2 / (th + sh + 10 ** (rin_db / 10) * LB["photocurrent_A"] ** 2 * of.BANDWIDTH)
    base = of.OAMChannel()
    ch_ind = of.OAMChannel(correlated=False)
    scint_variants = {"point_receiver": of.OAMChannel(point_receiver=True), "aperture_0.2m": of.OAMChannel(D_scint=of.D_R),
                      "shared_pm": of.OAMChannel(shared_pm=True)}
    disp = (0.05, 0.1, 0.2)
    for s2 in s2_list:
        t0 = time.time()
        rz, rs, rd = np.random.default_rng(seed), np.random.default_rng(seed + 1), np.random.default_rng(seed + 2)
        rv = {k: np.random.default_rng(seed + 1) for k in scint_variants}
        keys = (["baseline", "signal_dependent_shot_noise", "rin_-155dB", "ian_receiver", "independent_coeffs"]
                + list(scint_variants) + [f"displacement_{f}w" for f in disp])
        ind = {k: np.zeros(N, dtype=bool) for k in keys}
        for i in range(0, N, batch):
            b = min(batch, N - i)
            z = rz.standard_normal((b, base.j_list.size))
            I0 = base.scint(rs, b, s2)
            dv = rd.standard_normal((b, 2))
            Pc = base.power_coupling(base.coeffs(z, s2))
            G0 = Pc * I0[:, None, :]
            sl = slice(i, i + b)
            ind["baseline"][sl] = (of.sinr_mmse(G0, SNR0) < GTH).any(1)
            ind["signal_dependent_shot_noise"][sl] = (sinr_mmse_sd(G0, np.ones(base.M)) < GTH).any(1)
            ind["rin_-155dB"][sl] = (of.sinr_mmse(G0, snr_rin) < GTH).any(1)
            ind["ian_receiver"][sl] = (of.sinr_ian(G0, SNR0) < GTH).any(1)
            G = ch_ind.power_coupling(ch_ind.coeffs(z, s2)) * I0[:, None, :]
            ind["independent_coeffs"][sl] = (of.sinr_mmse(G, SNR0) < GTH).any(1)
            for k, ch in scint_variants.items():
                G = Pc * ch.scint(rv[k], b, s2)[:, None, :]
                ind[k][sl] = (of.sinr_mmse(G, SNR0) < GTH).any(1)
            for f in disp:
                G = coupling_displaced(base, z, s2, dv * f * base.w) * I0[:, None, :]
                ind[f"displacement_{f}w"][sl] = (of.sinr_mmse(G, SNR0) < GTH).any(1)
        rows = {}
        for k, v in ind.items():
            dlt = v.astype(float) - ind["baseline"].astype(float)
            rows[k] = {"outage": float(v.mean()), "count": int(v.sum()), "ci95": binom_ci(int(v.sum()), N),
                       "diff": float(dlt.mean()), "diff_ci95": 1.96 * float(dlt.std(ddof=1)) / math.sqrt(N)}
            print(f"sens s2={s2} {k}: outage {v.mean():.4g} (diff {dlt.mean():+.4g} +- {rows[k]['diff_ci95']:.2g})", flush=True)
        save(f"sens_{s2}", {"N": N, "s2": s2, "snr0_dB": LB["snr0_dB"], "rin_dB_per_Hz": rin_db,
                            "snr_with_rin_dB": 10 * math.log10(snr_rin), "displacement_fractions_of_w": list(disp),
                            "w_cm": 100 * base.w, "rows": rows})
        print(f"sens s2={s2} done in {time.time()-t0:.0f}s", flush=True)


# =============================================================================== power allocation
def stage_power(s2_list, N_fit=1_000_000, N_eval=1_000_000, seed=404):
    """Optical power allocation. kappa_n = P_n / (P_tot/M) is the optical power relative to uniform;
    with IM/DD the electrical signal power is kappa_n^2 * SNR0, and the constraint is sum(kappa_n) = M."""
    ch = of.OAMChannel()
    S_unif = SNR0
    for s2 in s2_list:
        t0 = time.time()
        G_fit = draw_many(ch, np.random.default_rng(seed), N_fit, s2)
        g = mmse_chunks(G_fit, S_unif)
        del G_fit
        fits = [of.fit_tail_gg(g[:, m])[:3] for m in range(ch.M)]
        eta = np.array([gb / S_unif for (_, _, gb) in fits])          # gbar'_m = eta_m * S_m, S_m = kappa_m^2 SNR0

        def outage_ub(kappa):
            return sum(of.gg_cdf_fast(math.sqrt(GTH / (eta[m] * kappa[m] ** 2 * S_unif)), fits[m][0], fits[m][1]) for m in range(ch.M))

        def grad_logk(kappa):
            # x_m = sqrt(gth/(eta kappa^2 S)) ~ kappa^{-1}  =>  d F(x_m)/d log kappa_m = -x_m f_I(x_m)
            gr = np.empty(ch.M)
            for m in range(ch.M):
                x = math.sqrt(GTH / (eta[m] * kappa[m] ** 2 * S_unif))
                gr[m] = -x * of.gg_pdf_I(x, fits[m][0], fits[m][1])
            return gr

        f_log = lambda u: math.log(outage_ub(np.exp(u)))
        g_log = lambda u: grad_logk(np.exp(u)) / outage_ub(np.exp(u))
        cons = {"type": "eq", "fun": lambda u: np.sum(np.exp(u)) - ch.M, "jac": lambda u: np.exp(u)}
        u0 = np.zeros(ch.M)
        sol = optimize.minimize(f_log, u0, jac=g_log, constraints=[cons], method="SLSQP",
                                bounds=[(-3, 2)] * ch.M, options={"maxiter": 500, "ftol": 1e-12})
        k_out = np.exp(sol.x)

        def neg_cap(u):
            k = np.exp(u)
            return -sum(ergodic_per_mode(eta[m] * k[m] ** 2 * S_unif, fits[m][0], fits[m][1]) for m in range(ch.M))

        sol2 = optimize.minimize(neg_cap, u0, constraints=[cons], method="SLSQP", bounds=[(-3, 2)] * ch.M,
                                 options={"maxiter": 300, "ftol": 1e-12})
        k_cap = np.exp(sol2.x)
        case_opt = {"outage_min_success": bool(sol.success), "outage_min_msg": str(sol.message), "outage_min_nit": int(sol.nit),
                    "capacity_max_success": bool(sol2.success), "capacity_max_msg": str(sol2.message)}
        G_ev = draw_many(ch, np.random.default_rng(seed + 1), N_eval, s2)
        case = {"N_fit": N_fit, "N_eval": N_eval, "s2": s2, "eta_dB": (10 * np.log10(eta)).tolist(),
                "fits": [list(f) for f in fits], "allocations": {}, "optimizer": case_opt}
        ind = {}
        for name, k in (("uniform", np.ones(ch.M)), ("outage_min", k_out), ("capacity_max", k_cap)):
            gg = mmse_chunks(G_ev, k ** 2 * S_unif)
            ind[name] = (gg < GTH).any(1)
            cnt = int(ind[name].sum())
            c = np.log2(1 + gg).sum(1)
            cnt_sd = int((sinr_mmse_sd(G_ev, k) < GTH).any(1).sum())
            case["allocations"][name] = {
                "kappa": k.tolist(), "P_mW": (1e3 * LB["P_tot_W"] / ch.M * k).tolist(),
                "sys_outage": cnt / N_eval, "count": cnt, "ci95": binom_ci(cnt, N_eval),
                "sys_outage_signal_dep_noise": cnt_sd / N_eval, "count_signal_dep_noise": cnt_sd,
                "per_mode_outage": (gg < GTH).mean(0).tolist(),
                "mean_se": float(c.mean()), "ec_norm_s1": of.ec_norm_mc(c, 1.0), "ub_model": outage_ub(k)}
            print(f"power s2={s2} {name}: outage {cnt/N_eval:.4g} (signal-dep. noise {cnt_sd/N_eval:.4g}; model bound {outage_ub(k):.4g}), "
                  f"mean SE {c.mean():.2f}, P_mW = {np.round(1e3 * LB['P_tot_W'] / ch.M * k, 2)}", flush=True)
        d = ind["outage_min"].astype(float) - ind["uniform"].astype(float)
        case["paired_diff_outage_min_minus_uniform"] = {"mean": float(d.mean()), "ci95": 1.96 * float(d.std(ddof=1)) / math.sqrt(N_eval)}
        save(f"power_{s2}", case)
        print(f"power s2={s2} done in {time.time()-t0:.0f}s", flush=True)
        del G_ev


# =============================================================================== Fig. 2
def stage_fig2(s2_list, N=1_000_000, part=None, snr_max=80.0):
    """part = (i, n) evaluates only the SNR points with index % n == i (all parts use the same channel
    draws, so the parts can run as separate processes and are merged by merge_fig2)."""
    ch = of.OAMChannel()
    snr_all = np.arange(30.0, snr_max + 2.5, 2.5)
    snr_grid = snr_all if part is None else snr_all[part[0]::part[1]]
    for s2 in s2_list:
        t0 = time.time()
        G = draw_many(ch, np.random.default_rng({0.2: 1101, 1.0: 1102, 2.5: 1103}[s2]), N, s2)
        a0, b0 = ch.gg_params(s2); ratio = a0 / b0
        rows = []
        for snr_dB in snr_grid:
            P = 10 ** (snr_dB / 10)
            g = mmse_chunks(G, P)
            out_m = (g < GTH)
            sys_cnt = int(out_m.any(1).sum())
            per_mom, per_tail = [], []
            for m in range(ch.M):
                a, b, gb = of.fit_equivalent_gg(g[:, m], ratio)
                per_mom.append(of.gg_cdf_fast(math.sqrt(GTH / gb), a, b))
                at, bt, gt, _, _ = of.fit_tail_gg(g[:, m])
                per_tail.append(of.gg_cdf_fast(math.sqrt(GTH / gt), at, bt))
            per_mom = np.array(per_mom); per_tail = np.array(per_tail)
            gi = of.sinr_ian(G[:200000].astype(np.float64), P)
            rows.append({"snr_dB": float(snr_dB), "mc_sys": sys_cnt / N, "mc_sys_count": sys_cnt, "mc_ci95": binom_ci(sys_cnt, N),
                         "mc_per_mode": out_m.mean(0).tolist(),
                         "ub_tail": float(min(1.0, per_tail.sum())), "product_tail": float(1 - np.prod(1 - per_tail)),
                         "ub_moment": float(min(1.0, per_mom.sum())),
                         "tail_per_mode": per_tail.tolist(), "moment_per_mode": per_mom.tolist(),
                         "ian_sys": float((gi < GTH).any(1).mean())})
            print(f"fig2 s2={s2} {snr_dB:.1f} dB: MC {sys_cnt/N:.4g}, tail UB {rows[-1]['ub_tail']:.4g}, moment UB {rows[-1]['ub_moment']:.3g}", flush=True)
        name = f"fig2_{s2}" if part is None else f"fig2_{s2}_part{part[0]}of{part[1]}"
        save(name, {"N": N, "s2": s2, "snr_dB": snr_grid.tolist(), "gamma_th_dB": 10 * math.log10(GTH), "rows": rows})
        print(f"fig2 s2={s2} done in {time.time()-t0:.0f}s", flush=True)
        del G


def merge_fig2(s2_list):
    import glob
    for s2 in s2_list:
        parts = sorted(glob.glob(os.path.join(OUT, f"fig2_{s2}_part*of*.json")))
        if not parts:
            continue
        ds = [json.load(open(p)) for p in parts]
        rows = sorted([r for d in ds for r in d["rows"]], key=lambda r: r["snr_dB"])
        save(f"fig2_{s2}", {"N": ds[0]["N"], "s2": s2, "snr_dB": [r["snr_dB"] for r in rows],
                            "gamma_th_dB": ds[0]["gamma_th_dB"], "rows": rows, "parts": len(parts)})
        print(f"merged {len(parts)} parts for s2={s2}: {len(rows)} SNR points", flush=True)


def plot_fig2():
    from matplotlib.lines import Line2D
    fig, ax = plt.subplots(figsize=(4.8, 4.1))
    for s2 in S2_MAIN:
        if not have(f"fig2_{s2}"):
            continue
        rows = load(f"fig2_{s2}")["rows"]; c = COLORS[s2]
        x = [r["snr_dB"] for r in rows]
        ax.semilogy(x, [max(r["ub_tail"], 1e-12) for r in rows], color=c, lw=1.6)
        ax.semilogy(x, [max(r["ub_moment"], 1e-60) for r in rows], color=c, lw=1.0, ls="--")
        xm = [r["snr_dB"] for r in rows if r["mc_sys_count"] >= 10]
        ym = [r["mc_sys"] for r in rows if r["mc_sys_count"] >= 10]
        ax.semilogy(xm, ym, ls="none", marker="o", ms=4.5, mfc="white", mec=c, mew=1.3)
    ax.axvline(LB["snr0_dB"], color="#999999", lw=0.8, ls=":")
    ax.set_ylim(1e-6, 1.5); ax.set_xlim(30, 80)
    ax.set_xlabel(r"Reference electrical SNR per mode, $\gamma_0$ (dB)"); ax.set_ylabel("System outage probability")
    style(ax)
    h = [Line2D([], [], color=COLORS[s2], lw=2.0) for s2 in S2_MAIN]
    h += [Line2D([], [], color="#555555", lw=1.6), Line2D([], [], color="#555555", lw=1.0, ls="--"),
          Line2D([], [], ls="none", marker="o", ms=4.5, mfc="white", mec="#555555", mew=1.3)]
    lab = [f"{NAMES[s2]} ($\\sigma_R^2={s2}$)" for s2 in S2_MAIN] + ["Union bound, tail-matched", "Union bound, moment-matched", "Monte Carlo"]
    ax.legend(h, lab, frameon=False, fontsize=7, ncol=2, loc="upper center", bbox_to_anchor=(0.45, -0.17),
              columnspacing=1.2, handlelength=2.2)
    fig.tight_layout(); fig.savefig(os.path.join(FIGDIR, "fig2_outage_mmse.png"), dpi=300, bbox_inches="tight")


# =============================================================================== operating point
def ks_kl_tail(g, a, b, gb, n_grid=400):
    """KS distance and KL divergence on a log-gamma grid, plus lower-tail checks."""
    gs = np.sort(g)
    n = gs.size
    lo, hi = np.log10(max(gs[0], 1e-12)), np.log10(gs[-1])
    grid = np.logspace(lo, hi, n_grid)
    F_emp = np.searchsorted(gs, grid, side="right") / n
    F_sur = of.gg_cdf_grid(np.sqrt(grid / gb), a, b)
    ks = float(np.max(np.abs(F_emp - F_sur)))
    pe = np.diff(np.concatenate([[0], F_emp, [1]])); ps_ = np.diff(np.concatenate([[0], F_sur, [1]]))
    mask = (pe > 0) & (ps_ > 0)
    kl = float(np.sum(pe[mask] * np.log(pe[mask] / ps_[mask])))
    tail = []
    for p in (1e-2, 1e-3, 1e-4):
        xq = gb * of.gg_quantile_I(p, a, b) ** 2
        cnt = int(np.searchsorted(gs, xq, side="left"))
        fe = cnt / n
        tail.append({"p": p, "x_dB": 10 * math.log10(xq), "F_emp": fe, "count": cnt,
                     "ratio": fe / p, "ci95_rel": 1.96 * math.sqrt((1 - fe) / max(cnt, 1))})
    return ks, kl, tail


def stage_oper(s2_list, N=10_000_000):
    ch = of.OAMChannel()
    P = SNR0
    for s2 in s2_list:
        t0 = time.time()
        rng = np.random.default_rng({0.2: 2201, 1.0: 2202, 2.5: 2203}[s2])
        a0, b0 = ch.gg_params(s2); ratio = a0 / b0
        gm = np.empty((N, ch.M), dtype=np.float32)
        ian_sys = 0; ian_med = []; Pc_sum = np.zeros((ch.M, ch.M)); n_pc = 0
        for i in range(0, N, 100000):
            b = min(100000, N - i)
            Gs = []
            for j in range(0, b, 2000):
                G, Pc, _ = ch.draw(rng, min(2000, b - j), s2, return_parts=True)
                Pc_sum += Pc.sum(0); n_pc += Pc.shape[0]
                Gs.append(G)
            G = np.concatenate(Gs)
            gm[i:i + b] = of.sinr_mmse(G, P)
            gi = of.sinr_ian(G, P)
            ian_sys += int((gi < GTH).any(1).sum())
            if i < 1_000_000:
                ian_med.append(gi)
        gi_all = np.concatenate(ian_med)
        sys_cnt = int((gm < GTH).any(1).sum())
        per_mode = []
        qq = {}
        for m in range(ch.M):
            gmm = gm[:, m].astype(np.float64)
            a, b, gb = of.fit_equivalent_gg(gmm, ratio)
            at, bt, gt, fun, ps_used = of.fit_tail_gg(gmm)
            ks, kl, tail = ks_kl_tail(gmm, a, b, gb)
            kst, klt, tailt = ks_kl_tail(gmm, at, bt, gt)
            cnt_m = int((gmm < GTH).sum())
            per_mode.append({"m": int(ch.modes[m]),
                             "moment": {"alpha": a, "beta": b, "gbar_dB": 10 * math.log10(gb), "ks": ks, "kl": kl, "tail": tail,
                                        "outage": of.gg_cdf_fast(math.sqrt(GTH / gb), a, b)},
                             "tailfit": {"alpha": at, "beta": bt, "gbar_dB": 10 * math.log10(gt), "ks": kst, "kl": klt, "tail": tailt,
                                         "obj": fun, "probs": ps_used, "outage": of.gg_cdf_fast(math.sqrt(GTH / gt), at, bt)},
                             "mc_outage": cnt_m / N, "mc_outage_count": cnt_m, "mc_outage_ci95": binom_ci(cnt_m, N),
                             "median_dB": float(np.median(10 * np.log10(gmm))),
                             "q_dB": {str(p): float(10 * np.log10(np.quantile(gmm, p))) for p in (1e-1, 1e-2, 1e-3, 1e-4, 1e-5)}})
            if int(ch.modes[m]) in (1, 4):
                pq = np.logspace(-5, math.log10(0.5), 30)
                emp = np.quantile(gmm, pq)
                sur = np.array([gb * of.gg_quantile_I(p, a, b) ** 2 for p in pq])
                surt = np.array([gt * of.gg_quantile_I(p, at, bt) ** 2 for p in pq])
                qq[str(int(ch.modes[m]))] = {"p": pq.tolist(), "emp_dB": (10 * np.log10(emp)).tolist(),
                                             "mom_dB": (10 * np.log10(sur)).tolist(), "tail_dB": (10 * np.log10(surt)).tolist()}
        ub = min(1.0, sum(pm["tailfit"]["outage"] for pm in per_mode))
        ub_mom = min(1.0, sum(pm["moment"]["outage"] for pm in per_mode))
        Pc_mean = Pc_sum / n_pc
        row = {"N": N, "s2": s2, "snr0_dB": LB["snr0_dB"], "J": of.J_MAX,
               "ian_median_dB": float(np.median(10 * np.log10(gi_all))), "ian_sys_outage": ian_sys / N,
               "ian_sys_ci95": binom_ci(ian_sys, N),
               "mmse_median_dB": float(np.median(10 * np.log10(gm))), "mmse_sys_outage": sys_cnt / N, "mmse_sys_count": sys_cnt,
               "mmse_sys_ci95": binom_ci(sys_cnt, N),
               "union_bound_tail": ub, "union_bound_moment": ub_mom, "per_mode": per_mode, "Pc_mean": Pc_mean.tolist(),
               "power_kept_in_set": Pc_mean.sum(0).tolist(), "qq": qq}
        save(f"oper_{s2}", row)
        print(f"oper s2={s2} done in {time.time()-t0:.0f}s: MMSE outage {sys_cnt/N:.4g}, IaN outage {ian_sys/N:.4g}, "
              f"tail UB {ub:.4g}", flush=True)
        del gm


def plot_oper():
    if have("oper_2.5"):
        Pc = np.array(load("oper_2.5")["Pc_mean"])
        fig, ax = plt.subplots(figsize=(3.9, 3.4))
        im = ax.imshow(10 * np.log10(Pc), cmap="Blues_r", vmin=-30, vmax=0)
        ax.set_xticks(range(8)); ax.set_yticks(range(8))
        ax.set_xticklabels([f"{m:+d}" for m in of.MODES]); ax.set_yticklabels([f"{m:+d}" for m in of.MODES])
        ax.set_xlabel("Transmitted mode $n$"); ax.set_ylabel("Received mode $m$")
        for i in range(8):
            for j in range(8):
                v = 10 * np.log10(Pc[i, j])
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=6.5, color="#222222" if v > -17 else "white")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cb.set_label(r"$\mathbb{E}[|h_{m,n}|^2]$ (dB)")
        fig.tight_layout(); fig.savefig(os.path.join(FIGDIR, "fig3_crosstalk_mean.png"), dpi=300)
    if not any(have(f"oper_{s2}") for s2 in S2_MAIN):
        return
    fig, axs = plt.subplots(1, 3, figsize=(7.16, 2.6), sharey=False)
    for ax, s2 in zip(axs, S2_MAIN):
        if not have(f"oper_{s2}"):
            continue
        qq = load(f"oper_{s2}")["qq"]
        for m, c in ((1, "#2a78d6"), (4, "#eb6834")):
            d = qq[str(m)]
            ax.semilogx(d["p"], d["emp_dB"], color=c, lw=1.6, label=f"$m=+{m}$: Monte Carlo")
            ax.semilogx(d["p"], d["tail_dB"], color=c, lw=0, marker="o", ms=3, mfc="white", mec=c, label=f"$m=+{m}$: tail-matched")
            ax.semilogx(d["p"], d["mom_dB"], color=c, lw=1.0, ls="--", label=f"$m=+{m}$: moment-matched")
        ax.axhline(10 * math.log10(GTH), color="#555555", lw=0.8, ls=":")
        ax.set_title(f"$\\sigma_R^2={s2}$", fontsize=9); style(ax)
        ax.set_xlabel("Probability $p$", fontsize=8)
    axs[0].set_ylabel("SINR quantile (dB)")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0.16, 1, 1)); fig.savefig(os.path.join(FIGDIR, "fig13_tail_quantiles.png"), dpi=300)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="analytics,jsweep,sens,power,fig2,oper,plot")
    ap.add_argument("--s2", default="0.2,1.0,2.5", help="Rytov variances for the Monte Carlo stages")
    ap.add_argument("--quick", action="store_true", help="small sample sizes for a smoke test")
    ap.add_argument("--part", default=None, help="i/n: evaluate a subset of the Fig. 2 SNR points")
    ap.add_argument("--snr-max", type=float, default=80.0, help="largest reference SNR of the Fig. 2 sweep (dB)")
    args = ap.parse_args()
    q = args.quick
    s2l = [float(x) for x in args.s2.split(",")]
    print(f"link budget: SNR0 = {LB['snr0_dB']:.2f} dB (thermal only {LB['snr_thermal_only_dB']:.2f} dB, shot only "
          f"{LB['snr_shot_only_dB']:.2f} dB), gamma_th = {10*math.log10(GTH):.2f} dB, J = {of.J_MAX}", flush=True)
    for st in args.stages.split(","):
        t0 = time.time()
        if st == "analytics":
            stage_analytics()
        elif st == "jsweep":
            stage_jsweep([s for s in s2l if s in (1.0, 2.5)], N=4_000 if q else 100_000)
        elif st == "sens":
            stage_sens([s for s in s2l if s in (1.0, 2.5)], N=4_000 if q else 200_000)
        elif st == "power":
            stage_power([s for s in s2l if s in (1.0, 2.5)], N_fit=10_000 if q else 1_000_000, N_eval=20_000 if q else 1_000_000)
        elif st == "fig2":
            part = None if args.part is None else tuple(int(x) for x in args.part.split("/"))
            stage_fig2(s2l, N=20_000 if q else 1_000_000, part=part, snr_max=args.snr_max)
        elif st == "fig2merge":
            merge_fig2(s2l)
        elif st == "oper":
            stage_oper(s2l, N=50_000 if q else 10_000_000)
        elif st == "plot":
            plot_jsweep(); plot_fig2(); plot_oper()
        print(f"stage {st} finished in {time.time()-t0:.0f}s", flush=True)
