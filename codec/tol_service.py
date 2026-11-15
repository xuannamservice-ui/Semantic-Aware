"""Service-level failed-slot rate (usage: python tol_service.py <dir with results/ of the codec> <phy_sim/results>) over the RIN x extra-loss grid (T6R11, sigma_R^2 = 2.5), with the Zernike
truncation correction. abar = pi_A + c_J * sum_K P(K) P(Psi < 30 | K); P(Psi < 30 | K) measured for K = 1, 2 on the
T6R11 traces (11 scenes, interleaved mapping, concealment network), K >= 3 counted as failed (upper bound)."""
import json, sys, glob
import numpy as np
R, P = sys.argv[1], sys.argv[2]
rows = []
for f in glob.glob(f"{R}/results/*/eval_layered_v2a_r1_T6R11_2.5.json"):
    e = json.load(open(f)); rows += [r for r in e["rows"] if r["policy"] == "interleaved" and r["method"] == "gen_fp"]
pk = {}
for K in (1, 2):
    x = [r["psnr_vs_full"] < 30 for r in rows if r["K"] == K]
    pk[K] = (np.mean(x) if x else 1.0, len(x))
print("P(Psi<30|K)", pk, "rows", len(rows))
cJ = 1.2721   # outage ratio J -> infinity / J = 435 (T6R11, sigma_R^2 = 2.5; jconv extrapolation)
tol = json.load(open(f"{P}/tolerance_2.5.json"))["configs"]["T6R11"]["grid"]
N = json.load(open(f"{P}/tolerance_2.5.json"))["N"]
out = []
for g in tol:
    h = np.array(g["nfail_hist"]) / N
    cond = h[1] * pk[1][0] + h[2] * pk[2][0] + h[3:].sum()
    ab = g["best_anchor_loss"] + cJ * cond
    out.append(dict(dL=g["extra_loss_dB"], rin=g["rin_dB_Hz"], pout=g["outage"], pout_J=cJ * g["outage"], abar=ab))
    print(f"dL={g['extra_loss_dB']:.2f} RIN={g['rin_dB_Hz']}: Pout={g['outage']:.3g} (J-corr {cJ * g['outage']:.3g}) abar={ab:.3g}")
json.dump(dict(pk={k: v[0] for k, v in pk.items()}, n={k: v[1] for k, v in pk.items()}, cJ=cJ, grid=out),
          open(f"{P}/tol_service_2.5.json", "w"), indent=1)
