"""Second channel: multi-beam FSO spatial multiplexing with PAM-4, Gamma-Gamma scintillation and a common pointing error.

W = 8 beams, each carrying one stream to its own small aperture (point-receiver Gamma-Gamma parameters, oamfso.gg_point);
no OAM, no mode sorter and no equalizer. The large-scale factor X is common to all beams and the small-scale factors Y_w
are independent; a building sway displaces all beams together by a radial offset r with Rayleigh distribution
(standard deviation sway * w_eq per axis), which scales the received power by h_p = exp(-2 r^2 / w_eq^2) (Gaussian-beam
pointing loss with the aperture capture absorbed in the link budget). Link budget per beam as for eight OAM streams
(run_modeset.snr0(8)), PAM-4 with the decision-SNR penalty (L - 1)^2 = 9.5 dB. A beam fails in a slot if
snr0(8) (h_p X Y_w)^2 / 9 < gamma_th. Output, in the format of run_modeset.py: results/modeset_mb.json (configurations MB8M
at sigma_R^2 = 1.0 and MB8S at 2.5, sway 0.4) and results/traces_mb.npz (failure patterns of the first 1e6 slots).
usage: python run_multibeam.py 10000000
"""
import json
import math
import os
import sys

import numpy as np

import oamfso as of
import run_modeset as rm

HERE = os.path.dirname(os.path.abspath(__file__))
W = 8; L = 4; SWAY = 0.4; NTR = 1_000_000; CH = 1_000_000
CFGS = {"MB8M": (1.0, 9811), "MB8S": (2.5, 9812)}


def main(N):
    thr = math.sqrt(rm.GTH / rm.snr0(W)) * (L - 1)
    res = {"model": __doc__.split("\n")[0], "beams": W, "pam_levels": L, "sway_over_weq": SWAY, "N": N, "configs": {}}
    traces = {}
    w = (1 << np.arange(W)).astype(np.uint16)
    for cfg, (s2, seed) in CFGS.items():
        a, b = of.gg_point(s2)
        rng = np.random.default_rng(seed)
        nfail = np.zeros(W + 1, np.int64); per = np.zeros(W, np.int64); pair = np.zeros((W, W), np.int64); tr = []
        done = 0
        while done < N:
            n = min(CH, N - done)
            X = rng.gamma(a, 1 / a, n); Y = rng.gamma(b, 1 / b, (n, W))
            r2 = (rng.standard_normal(n) ** 2 + rng.standard_normal(n) ** 2) * SWAY ** 2
            f = (np.exp(-2 * r2)[:, None] * X[:, None] * Y) < thr
            nfail += np.bincount(f.sum(1), minlength=W + 1); per += f.sum(0)
            fi = f.astype(np.int64); pair += fi.T @ fi
            if done < NTR:
                tr.append((f[:NTR - done].astype(np.uint16) @ w).astype(np.uint8))
            done += n
        traces[cfg] = np.concatenate(tr)[:NTR]
        iu = [(i, j) for i in range(W) for j in range(i + 1, W)]; best = min(iu, key=lambda ij: pair[ij])
        res["configs"][cfg] = {"s2": s2, "alpha": a, "beta": b, "tx": list(range(1, W + 1)), "snr_labels": ["op"],
                               "outage_count": [int(N - nfail[0])], "per_mode_count": [per.tolist()], "nfail": nfail.tolist(),
                               "outage": float(1 - nfail[0] / N), "best_anchor_pair": [best[0] + 1, best[1] + 1],
                               "best_anchor_loss": float(pair[best] / N), "pair_joint_fail": (pair / N).tolist()}
        o = res["configs"][cfg]
        print(cfg, f"s2={s2}: outage {o['outage']:.3e}, K hist {nfail[1:6].tolist()}, anchor pair loss {o['best_anchor_loss']:.2e}", flush=True)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(res, open(os.path.join(HERE, "results", "modeset_mb.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(HERE, "results", "traces_mb.npz"), **traces)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 10_000_000)
