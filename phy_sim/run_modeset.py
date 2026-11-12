"""Transmit mode sets and mode-sorter port sets of the OAM-FSO link (Sections III-E, VII-C and VII-D).

Every configuration (transmitted LG_{0,l} modes, sorter ports) is evaluated on common random numbers:
the Zernike phase screen and the scintillation factor of each transmitted mode are shared, so the
differences between configurations are paired. Total optical transmit power 20 dBm is split uniformly
over the active modes; every port has the noise variance of a data port of the same configuration
(oamfso.link_budget(M)), which is conservative for ports that only collect crosstalk.

Per configuration and Rytov variance the stage records:
  * MMSE and ZF system outage (any active stream below the OOK HD-FEC threshold) and per-mode outage on a
    reference-SNR grid (diversity order, Fig. "diversity");
  * at the link-budget operating point: number of failed streams, histogram of the failure pattern (bit i
    set if the i-th transmitted mode fails), pairwise joint failures, loss of replicated anchor placements,
    mean sum SE, effective capacity of the sum SE and its histogram;
  * the failure pattern of the first 1e6 slots of every configuration (traces_<s2>.npz), for trace-driven
    evaluation of the concealment codec;
  * sorter insertion loss 1 dB and 2 dB above the nominal 3 dB for the oversampled sorters;
  * per-realization selection of the best 7 of the 8 modes (max-min MMSE SINR) with the square sorter.
Results: sim/results/modeset_<s2>.json (and modeset_point_<s2>.json with point-receiver scintillation).
"""
import argparse
import json
import math
import os
import time

import numpy as np

import oamfso as of

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
TX8 = (-4, -3, -2, -1, 1, 2, 3, 4)
CFG = {
    "T8R8": (TX8, TX8),
    "T8R9": (TX8, (-4, -3, -2, -1, 0, 1, 2, 3, 4)),
    "T8R10": (TX8, (-5, -4, -3, -2, -1, 1, 2, 3, 4, 5)),
    "T8R11": (TX8, tuple(range(-5, 6))),
    "T8R13": (TX8, tuple(range(-6, 7))),
    "T7R8": ((-3, -2, -1, 1, 2, 3, 4), TX8),
    "T6R6": ((-3, -2, -1, 1, 2, 3), (-3, -2, -1, 1, 2, 3)),
    "T6R8": ((-3, -2, -1, 1, 2, 3), TX8),
    "T6R11": ((-3, -2, -1, 1, 2, 3), tuple(range(-5, 6))),
    "T4R8": ((-2, -1, 1, 2), TX8),
}
SEL = "S7R8"                       # best 7 of TX8 per realization, square sorter
RX_ALL = tuple(range(-6, 7))
SNR_GRID_DB = tuple(range(30, 101, 5))
EXTRA_LOSS_DB = (1.0, 2.0)          # additional sorter insertion loss for N_r > M
EC_S = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
SE_BIN = 0.25                       # bit/s/Hz, histogram of the sum SE
ANCHORS = {"+1": (1,), "+-1": (-1, 1), "+-2": (-2, 2), "+-1,+-2": (-2, -1, 1, 2), "+-1..+-3": (-3, -2, -1, 1, 2, 3)}
GTH = of.gamma_th_from_ber()
SEEDS = {0.2: 3301, 0.5: 3305, 1.0: 3302, 1.5: 3306, 2.5: 3303}


def snr0(M, extra_loss_db=0.0, rin_db_hz=None):
    """Link-budget reference SNR per mode (oamfso.link_budget) with an optional extra optical loss and an
    optional laser relative intensity noise (variance RIN * B * i_p^2)."""
    p_tot = 1e-3 * 10 ** (of.P_TOT_DBM / 10)
    loss_db = of.ATM_LOSS_DB_PER_KM * of.L_LINK / 1000 + of.DEMUX_LOSS_DB + extra_loss_db
    i_p = of.RESPONSIVITY * p_tot / M * 10 ** (-loss_db / 10)
    s_th = 4 * of.KB * of.T_K * 10 ** (of.F_N_DB / 10) * of.BANDWIDTH / of.R_LOAD
    s_sh = 2 * of.Q_E * i_p * of.BANDWIDTH
    s_rin = 0.0 if rin_db_hz is None else 10 ** (rin_db_hz / 10) * of.BANDWIDTH * i_p ** 2
    return i_p ** 2 / (s_th + s_sh + s_rin)


class LSE:
    """Streaming log-mean-exp of -s*c for several s (effective capacity -(1/s) ln E exp(-s c))."""

    def __init__(self, s_list):
        self.s = np.asarray(s_list, dtype=float)
        self.mx = np.full(self.s.size, -np.inf)
        self.sm = np.zeros(self.s.size)
        self.n = 0

    def add(self, c):
        x = -self.s[:, None] * c[None, :]
        bx = x.max(1)
        new = np.maximum(self.mx, bx)
        self.sm = self.sm * np.exp(self.mx - new) + np.exp(x - new[:, None]).sum(1)
        self.mx = new
        self.n += c.size

    def ec(self):
        return (-(self.mx + np.log(self.sm / self.n)) / self.s).tolist()


def run(s2, N, batch, point, store_dir, tag, extra_db=(), suffix="", rin=None):
    t0 = time.time()
    base = of.OAMChannel(point_receiver=point)
    gr = base.grid
    U = {m: of.lg_mode(m, gr["r"], gr["phi"], base.w) for m in RX_ALL}
    Ur = np.stack([U[m] for m in RX_ALL]); Ut = np.stack([U[m] for m in TX8])
    Q = (np.conj(Ur)[:, None, :] * Ut[None, :, :] * gr["dA"]).reshape(len(RX_ALL) * len(TX8), -1).T
    Qr, Qi = np.ascontiguousarray(Q.real), np.ascontiguousarray(Q.imag)
    ridx = {k: [RX_ALL.index(m) for m in rx] for k, (tx, rx) in CFG.items()}
    tidx = {k: [TX8.index(m) for m in tx] for k, (tx, rx) in CFG.items()}

    # SNR list per configuration: grid, operating point, extra sorter loss (oversampled sorters only)
    snrs, labels = {}, {}
    for k, (tx, rx) in list(CFG.items()) + [(SEL, (TX8, TX8))]:
        M = 7 if k == SEL else len(tx)
        lst = [10 ** (d / 10) for d in SNR_GRID_DB] + [snr0(M)]
        lab = [f"{d}dB" for d in SNR_GRID_DB] + ["op"]
        for e in extra_db:
            lst.append(10 ** (e / 10)); lab.append(f"x{e:g}dB")
        if rin is not None:
            lst.append(snr0(M, 0.0, rin)); lab.append("op_rin")
        if len(rx) > len(tx):
            for e in EXTRA_LOSS_DB:
                lst.append(snr0(M, e)); lab.append(f"op+{e:g}dB")
        snrs[k] = np.array(lst); labels[k] = lab

    acc = {}
    for k, (tx, rx) in CFG.items():
        M, ns = len(tx), len(snrs[k])
        acc[k] = {"out": np.zeros(ns, np.int64), "out_zf": np.zeros(ns, np.int64), "pm": np.zeros((ns, M), np.int64),
                  "nfail_grid": np.zeros((ns, M + 1), np.int64),
                  "nfail": np.zeros(M + 1, np.int64), "pair": np.zeros((M, M), np.int64),
                  "pattern": np.zeros(2 ** M, np.int64),
                  "anchor": {a: 0 for a, mm in ANCHORS.items() if set(mm) <= set(tx)},
                  "se": 0.0, "lse": LSE(EC_S), "se_hist": np.zeros(int(200 / SE_BIN), np.int64)}
    nsel = len(snrs[SEL])
    sel = {"out": np.zeros(nsel, np.int64), "nfail": np.zeros(8, np.int64), "drop": np.zeros(8, np.int64),
           "se": 0.0, "lse": LSE(EC_S), "se_hist": np.zeros(int(200 / SE_BIN), np.int64)}
    store = {}
    n_store = min(N, 1_000_000) if store_dir else 0
    n_trace = min(N, 1_000_000)
    traces = {k: np.zeros(n_trace, np.uint8) for k in CFG}
    if n_store:
        for k in ("T8R8", "T8R11", "T6R8"):
            store[k] = {"mmse": np.empty((n_store, len(CFG[k][0])), np.float32),
                        "zf": np.empty((n_store, len(CFG[k][0])), np.float32)}

    rng = np.random.default_rng(SEEDS[s2] + (1000 if point else 0))
    done = 0
    while done < N:
        b = min(batch, N - done)
        z = rng.standard_normal((b, base.j_list.size))
        Iall = base.scint(rng, b, s2)                         # (b, 8): one factor per mode of TX8
        Phi = base.coeffs(z, s2) @ base.Z
        c, s = np.cos(Phi), np.sin(Phi)
        hr = c @ Qr - s @ Qi
        hi = c @ Qi + s @ Qr
        Gall = (hr ** 2 + hi ** 2).reshape(b, len(RX_ALL), len(TX8)) * Iall[:, None, :]
        del Phi, c, s, hr, hi
        for k, (tx, rx) in CFG.items():
            a = acc[k]; M = len(tx)
            G = Gall[:, ridx[k]][:, :, tidx[k]]
            lam, V2 = of.gram_eig(G)
            zf = of.zf_gain_eig(lam, V2)
            iop = labels[k].index("op")
            for i, snr in enumerate(snrs[k]):
                g = of.sinr_mmse_eig(lam, V2, snr)
                fail = g < GTH
                a["out"][i] += int(fail.any(1).sum())
                a["out_zf"][i] += int((snr * zf < GTH).any(1).sum())
                a["pm"][i] += fail.sum(0)
                a["nfail_grid"][i] += np.bincount(fail.sum(1), minlength=M + 1)
                if i == iop:
                    a["nfail"] += np.bincount(fail.sum(1), minlength=M + 1)
                    fi = fail.astype(np.int64)
                    a["pair"] += fi.T @ fi
                    mask = fi @ (1 << np.arange(M))
                    a["pattern"] += np.bincount(mask, minlength=2 ** M)
                    if n_trace and done < n_trace:
                        e = min(b, n_trace - done)
                        traces[k][done:done + e] = mask[:e]
                    for an, mm in ANCHORS.items():
                        if an in a["anchor"]:
                            a["anchor"][an] += int(fail[:, [tx.index(m) for m in mm]].all(1).sum())
                    cse = np.log2(1 + g).sum(1)
                    a["se"] += float(cse.sum()); a["lse"].add(cse)
                    a["se_hist"] += np.bincount(np.minimum((cse / SE_BIN).astype(int), a["se_hist"].size - 1),
                                                minlength=a["se_hist"].size)
                    if k in store and done < n_store:
                        e = min(b, n_store - done)
                        store[k]["mmse"][done:done + e] = g[:e]; store[k]["zf"][done:done + e] = (snr * zf)[:e]
        # per-realization choice of the best 7 of 8 modes (max-min MMSE SINR), square sorter
        G8 = Gall[:, ridx["T8R8"]]
        eig7 = [of.gram_eig(np.delete(G8, j, axis=2)) for j in range(8)]
        iop = labels[SEL].index("op")
        for i, snr in enumerate(snrs[SEL]):
            gs = [of.sinr_mmse_eig(l7, v7, snr) for (l7, v7) in eig7]
            mins = np.stack([g.min(1) for g in gs], axis=1)          # (b, 8): min SINR when mode j is dropped
            best = mins.argmax(1)
            sel["out"][i] += int((mins.max(1) < GTH).sum())
            if i == iop:
                gb = np.stack(gs, axis=1)[np.arange(b), best]          # (b, 7)
                sel["nfail"] += np.bincount((gb < GTH).sum(1), minlength=8)
                sel["drop"] += np.bincount(best, minlength=8)
                cse = np.log2(1 + gb).sum(1)
                sel["se"] += float(cse.sum()); sel["lse"].add(cse)
                sel["se_hist"] += np.bincount(np.minimum((cse / SE_BIN).astype(int), sel["se_hist"].size - 1),
                                              minlength=sel["se_hist"].size)
        done += b
        if done % (batch * 250) == 0 or done == N:
            print(f"{tag} s2={s2}: {done}/{N} samples, {time.time()-t0:.0f}s, T8R8 op outage "
                  f"{acc['T8R8']['out'][labels['T8R8'].index('op')]/done:.4g}", flush=True)

    al, be = base.gg_params(s2)
    res = {"s2": s2, "N": N, "point_receiver": point, "gamma_th_dB": 10 * math.log10(GTH), "gg_alpha": al, "gg_beta": be,
           "snr_grid_dB": list(SNR_GRID_DB), "ec_s": list(EC_S), "se_bin": SE_BIN, "configs": {}}
    for k, (tx, rx) in CFG.items():
        a = acc[k]; iop = labels[k].index("op")
        res["configs"][k] = {
            "tx": list(tx), "rx": list(rx), "M": len(tx), "Nr": len(rx), "d_pred": 0.5 * min(len(rx) - len(tx) + 1, al, be),
            "snr_labels": labels[k], "snr_dB": (10 * np.log10(snrs[k])).tolist(),
            "outage_count": a["out"].tolist(), "outage_zf_count": a["out_zf"].tolist(), "per_mode_count": a["pm"].tolist(),
            "nfail_grid": a["nfail_grid"].tolist(),
            "op": {"snr_dB": 10 * math.log10(snrs[k][iop]), "outage": a["out"][iop] / N, "count": int(a["out"][iop]),
                   "nfail_hist": a["nfail"].tolist(), "pair_count": a["pair"].tolist(), "anchor_count": a["anchor"],
                   "pattern_count": {int(i): int(c) for i, c in enumerate(a["pattern"]) if c},
                   "mean_se": a["se"] / N, "ec": a["lse"].ec(), "se_hist": a["se_hist"].tolist()}}
    iop = labels[SEL].index("op")
    res["configs"][SEL] = {"tx": list(TX8), "rx": list(TX8), "M": 7, "Nr": 8, "select": "best 7 of 8, max-min MMSE SINR",
                           "snr_labels": labels[SEL], "snr_dB": (10 * np.log10(snrs[SEL])).tolist(),
                           "outage_count": sel["out"].tolist(),
                           "op": {"snr_dB": 10 * math.log10(snrs[SEL][iop]), "outage": sel["out"][iop] / N,
                                  "count": int(sel["out"][iop]), "nfail_hist": sel["nfail"].tolist(),
                                  "dropped_mode_hist": sel["drop"].tolist(), "mean_se": sel["se"] / N, "ec": sel["lse"].ec(),
                                  "se_hist": sel["se_hist"].tolist()}}
    res["runtime_s"] = time.time() - t0
    name = (f"modeset_point_{s2}" if point else f"modeset_{s2}") + suffix
    with open(os.path.join(OUT, name + ".json"), "w") as f:
        json.dump(res, f, indent=1, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else float(o))
    np.savez_compressed(os.path.join(OUT, f"traces_{'point_' if point else ''}{s2}{suffix}.npz"),
                        **{k: v for k, v in traces.items()}, modes=np.array(TX8))
    if n_store:
        for k, d in store.items():
            np.savez_compressed(os.path.join(store_dir, f"sinr_{k}_{s2}{'_point' if point else ''}.npz"), **d,
                                snr=snrs[k][labels[k].index("op")])
    print(f"{tag} s2={s2} done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--s2", type=float, required=True)
    ap.add_argument("--N", type=int, default=10_000_000)
    ap.add_argument("--batch", type=int, default=2000)
    ap.add_argument("--point", action="store_true", help="point-receiver Gamma-Gamma parameters (scintillation cap check)")
    ap.add_argument("--store", default="", help="directory for per-mode SINR samples of T8R8, T8R11 and T6R8 (first 1e6)")
    ap.add_argument("--tag", default="modeset")
    ap.add_argument("--extra-db", default="", help="comma-separated additional reference SNR values (dB)")
    ap.add_argument("--suffix", default="", help="suffix of the result file name")
    ap.add_argument("--rin", type=float, default=None, help="add the operating point with laser RIN (dB/Hz)")
    a = ap.parse_args()
    extra = tuple(float(x) for x in a.extra_db.split(",") if x)
    run(a.s2, a.N, a.batch, a.point, a.store, a.tag, extra, a.suffix, a.rin)
