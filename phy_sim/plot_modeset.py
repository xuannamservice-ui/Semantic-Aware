"""Summary statistics and figure of the mode-set / sorter-port stage (run_modeset.py).

Writes sim/results/modeset_summary.json and the manuscript figure fig2_diversity.png.
"""
import itertools
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
FIGDIR = os.path.dirname(HERE)
S2 = (0.2, 1.0, 2.5)
NAMES = {0.2: "Weak", 1.0: "Moderate", 2.5: "Strong"}
SHOW = [("T8R8", "#2a78d6", "o"), ("T8R9", "#eb6834", "s"), ("T8R11", "#1baf7a", "^"),
        ("T6R8", "#e87ba4", "v"), ("T4R8", "#eda100", "D"), ("T6R11", "#008300", "P")]
LABEL = {"T8R8": "$M=8$, $N_r=8$", "T8R9": "$M=8$, $N_r=9$", "T8R10": "$M=8$, $N_r=10$", "T8R11": "$M=8$, $N_r=11$",
         "T8R13": "$M=8$, $N_r=13$", "T7R8": "$M=7$, $N_r=8$", "T6R6": "$M=6$, $N_r=6$", "T6R8": "$M=6$, $N_r=8$",
         "T6R11": "$M=6$, $N_r=11$", "T4R8": "$M=4$, $N_r=8$", "S7R8": "best 7 of 8, $N_r=8$"}
MIN_EV_PLOT, MIN_EV_SLOPE = 10, 50
FEC_OVERHEAD = 1.07           # OOK with 7% hard-decision FEC overhead: net rate B/1.07 per stream

plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.edgecolor": "#888", "axes.linewidth": 0.6})


def load(name):
    p = os.path.join(OUT, name + ".json")
    return json.load(open(p)) if os.path.exists(p) else None


def ci95(cnt, N):
    p = cnt / N
    return 1.96 * math.sqrt(max(p * (1 - p), 0.0) / N)


def local_slopes(x_db, cnt, N):
    """Decades of outage per decade of gamma0 between consecutive grid points, with a 95% CI (delta method)."""
    out = []
    for i in range(len(x_db) - 1):
        a, b = cnt[i], cnt[i + 1]
        if a >= MIN_EV_SLOPE and b >= MIN_EV_SLOPE:
            s = -(math.log10(b / N) - math.log10(a / N)) / ((x_db[i + 1] - x_db[i]) / 10)
            se = math.sqrt((1 - a / N) / a + (1 - b / N) / b) / math.log(10) / ((x_db[i + 1] - x_db[i]) / 10)
            out.append({"from_dB": x_db[i], "to_dB": x_db[i + 1], "slope": s, "ci95": 1.96 * se})
    return out


def summarize(r):
    N = r["N"]
    grid = r["snr_grid_dB"]
    ng = len(grid)
    res = {"N": N, "gg_alpha": r["gg_alpha"], "gg_beta": r["gg_beta"], "configs": {}}
    for k, c in r["configs"].items():
        op = c["op"]
        cnt = c["outage_count"]
        s = {"M": c["M"], "Nr": c["Nr"], "d_pred": c.get("d_pred"),
             "op_snr_dB": op["snr_dB"], "outage": op["outage"], "count": op["count"], "ci95": ci95(op["count"], N),
             "outage_grid": [x / N for x in cnt[:ng]], "count_grid": cnt[:ng],
             "slopes": local_slopes(grid, cnt[:ng], N), "mean_se": op["mean_se"],
             "ec": dict(zip([str(x) for x in r["ec_s"]], op["ec"]))}
        if s["slopes"]:
            s["slope_high"] = s["slopes"][-1]
        labs = c["snr_labels"]
        for e in ("op+1dB", "op+2dB"):
            if e in labs:
                s["outage_" + e.replace("+", "_plus_")] = cnt[labs.index(e)] / N
        if "outage_zf_count" in c:
            iop = labs.index("op")
            s["outage_zf"] = c["outage_zf_count"][iop] / N
            s["count_zf"] = c["outage_zf_count"][iop]
            s["outage_zf_grid"] = [x / N for x in c["outage_zf_count"][:ng]]
            s["slopes_zf"] = local_slopes(grid, c["outage_zf_count"][:ng], N)
            pmc = np.array(c["per_mode_count"][iop])
            s["per_mode"] = {str(m): pmc[i] / N for i, m in enumerate(c["tx"])}
            s["sum_per_mode"] = float(pmc.sum() / N)
            s["mean_lost_fraction"] = float(pmc.sum() / N / c["M"])
        nf = np.array(op["nfail_hist"], dtype=float)
        # on-off service of the OOK streams: n_ok = M - K streams of B/1.07 bit/s each (7% HD-FEC overhead)
        pk_all = nf / nf.sum()
        n_ok = len(nf) - 1 - np.arange(len(nf))
        s["mean_streams_delivered"] = float(np.dot(pk_all, n_ok))
        s["ec_onoff"] = {str(x): float(-math.log(np.dot(pk_all, np.exp(-x * n_ok / FEC_OVERHEAD))) / x) for x in r["ec_s"]}
        nout = nf[1:].sum()
        if nout > 0:
            s["p_k_given_outage"] = (nf[1:] / nout).tolist()
            s["mean_k_given_outage"] = float(np.dot(np.arange(1, len(nf)), nf[1:]) / nout)
            s["p_k"] = (nf / N).tolist()
        if "se_hist" in op:
            h = np.array(op["se_hist"], dtype=float)
            centers = (np.arange(h.size) + 0.5) * r["se_bin"]
            rng = np.random.default_rng(12345)
            boot = rng.multinomial(int(h.sum()), h / h.sum(), size=200)
            ci = {}
            for x in r["ec_s"]:
                w = np.exp(-x * (centers - centers.min()))
                ecs = centers.min() - np.log(boot @ w / h.sum()) / x
                lo, hi = np.percentile(ecs, [2.5, 97.5])
                ci[str(x)] = [float(lo), float(hi)]
            s["ec_boot_ci95"] = ci
            cdf = np.cumsum(h) / h.sum()
            s["se_quantile"] = {str(q): float(centers[np.searchsorted(cdf, q)]) for q in (1e-4, 1e-3, 1e-2)}
        if "pattern_count" in op:
            pat = {int(i): v for i, v in op["pattern_count"].items()}
            tx = c["tx"]
            M = len(tx)

            def p_all(S):
                mask = sum(1 << tx.index(m) for m in S)
                return sum(v for i, v in pat.items() if i & mask == mask) / N
            pairs = {f"{a},{b}": p_all((a, b)) for a, b in itertools.combinations(tx, 2)}
            best = min(pairs, key=pairs.get)
            s["anchor"] = {"single": {str(m): p_all((m,)) for m in tx}, "pairs": pairs, "best_pair": best,
                           "best_pair_loss": pairs[best], "count_best_pair": int(round(pairs[best] * N))}
            for name, S in (("+-1", (-1, 1)), ("+-2", (-2, 2)), ("+-1,+-2", (-2, -1, 1, 2))):
                if set(S) <= set(tx):
                    s["anchor"][name] = p_all(S)
            top = sorted(pat.items(), key=lambda kv: -kv[1])
            s["top_patterns"] = [{"failed": [tx[j] for j in range(M) if i >> j & 1], "prob": v / N} for i, v in top[:12]]
            s["n_patterns"] = len(pat)
        if "dropped_mode_hist" in op:
            s["dropped_mode_hist"] = op["dropped_mode_hist"]
        res["configs"][k] = s
    return res


def plot(summ):
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.9), sharex=True, gridspec_kw={"height_ratios": [1.6, 1.0]})
    for j, s2 in enumerate((1.0, 2.5)):
        sm = summ.get(str(s2))
        if sm is None:
            continue
        r = load(f"modeset_{s2}")
        grid = np.array(r["snr_grid_dB"], dtype=float)
        ax, axs = axes[0, j], axes[1, j]
        for k, col, mk in SHOW:
            c = sm["configs"][k]
            cnt = np.array(c["count_grid"])
            ok = cnt >= MIN_EV_PLOT
            ax.semilogy(grid[ok], np.array(c["outage_grid"])[ok], color=col, marker=mk, ms=3.5, lw=1.2, label=LABEL[k])
            if k == "T8R8":
                z = np.array(c["outage_zf_grid"]) * sm["N"]
                okz = z >= MIN_EV_PLOT
                ax.semilogy(grid[okz], np.array(c["outage_zf_grid"])[okz], color=col, ls="--", lw=1.0, label="$M=8$, $N_r=8$, ZF")
            sl = c["slopes"]
            if sl:
                xs = [0.5 * (q["from_dB"] + q["to_dB"]) for q in sl]
                axs.errorbar(xs, [q["slope"] for q in sl], yerr=[q["ci95"] for q in sl], color=col, marker=mk, ms=3.5,
                             lw=1.0, capsize=1.5)
            axs.axhline(c["d_pred"], color=col, lw=0.8, ls=":")
        for a in (ax, axs):
            a.axvline(sm["configs"]["T8R8"]["op_snr_dB"], color="#555", lw=0.7, ls=":")
            a.grid(color="#e5e5e5", lw=0.5)
            a.set_axisbelow(True)
            for sp in ("top", "right"):
                a.spines[sp].set_visible(False)
        ax.set_title(f"{NAMES[s2]} turbulence ($\\sigma_R^2 = {s2}$)", fontsize=9)
        ax.set_ylim(1e-7 * 0.5, 1.2)
        axs.set_ylim(0, 3.3)
        axs.set_xlabel("Reference SNR $\\gamma_0$ (dB)")
        if j == 0:
            ax.set_ylabel("System outage $P_{out}$")
            axs.set_ylabel("Local slope")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    fig.savefig(os.path.join(FIGDIR, "fig2_diversity.png"), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    summ = {}
    for s2 in S2:
        r = load(f"modeset_{s2}")
        if r is not None:
            summ[str(s2)] = summarize(r)
    rp = load("modeset_point_2.5")
    if rp is not None:
        summ["point_2.5"] = summarize(rp)
    with open(os.path.join(OUT, "modeset_summary.json"), "w") as f:
        json.dump(summ, f, indent=1)
    if "1.0" in summ or "2.5" in summ:
        plot(summ)
    for key, sm in summ.items():
        for k, c in sm["configs"].items():
            sh = c.get("slope_high")
            print(f"{key:9s} {k:6s} d={c['d_pred']} Pout={c['outage']:.3g} ({c['count']}) "
                  f"slope_hi={sh['slope']:.2f}±{sh['ci95']:.2f} @{sh['from_dB']}-{sh['to_dB']}" if sh else
                  f"{key:9s} {k:6s} d={c['d_pred']} Pout={c['outage']:.3g} ({c['count']}) no slope")
