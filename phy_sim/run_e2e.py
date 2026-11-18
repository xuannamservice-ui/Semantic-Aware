"""Codeword-level PHY trace for the end-to-end experiment (sigma_R^2 = 2.5, tau_c = 2.5 ms, J = 780).

One Gauss-Markov chain per slot as in run_aging.py: slot T_f = 1 ms, NPT = 11 channel points at 0, 0.1, ..., 1.0 ms
(spacing 100 us), Model-B step correlation rho(T_f)^(1/10) on the normalized Zernike vector (j = 4..780) and on the
Gaussian copulas of X and Y_n. Common random numbers across receiver variants and configurations (T8R8 is evaluated
on the first 1e6 slots of the T6R11 chain).
Receiver variants (post-MMSE SINR per stream at every point):
  paper         ideal CSI, no sorter crosstalk, channel of point 0 held for the whole slot (model of the paper)
  paper435      as paper with the leading J = 435 terms of the same screen (Cholesky factor is lower triangular)
  integ         channel evolves inside the slot; MMSE filter from a 64-symbol pilot estimate refreshed every 1 us
                (channel 0.5 us earlier, as pilot1us in run_aging.py); sorter crosstalk -20 dB (run_budget.py)
  integ_rin165  integ with laser RIN -165 dB/Hz (run_modeset.snr0)
  integ_rin160  integ with laser RIN -160 dB/Hz
  integ_jit     integ with residual pointing jitter: per-slot lateral displacement d of the received beam relative
                to the sorter axis, d Rayleigh with E|d|^2 = (0.1 w(L))^2, transmitted LG modes evaluated at r - d
                (pilots see the displaced channel); evaluated on the first NJIT = 2e6 slots, since it doubles the
                memory traffic per slot (bandwidth-bound host)
Pilot channel: the screen 0.5 us earlier is Phi_p = rp Phi + sqrt(1 - rp^2) (w @ L^T Z), w standard normal; the term
w @ L^T Z is taken from the screen of slot b+1+t of the same chunk at point t, which has the same distribution and is
independent of slot b and of the other points of slot b (saves one 777-term synthesis per point).
FEC at codeword level: each stream carries 1e7 OOK symbols per slot (10 GBd); the slot is split into NSEG = 100
segments of 10 us (1e5 bits, 49 RS(255,239) codewords); the SINR in dB is interpolated linearly between the points
and a segment fails if the SINR at its midpoint is below the threshold: G.975.1 (BER 3.8e-3), RS(255,239) at BER 1e-3
and 1.8e-4 (oamfso.gamma_th_from_ber).
Output (results/e2e/): e2e_summary.json, e2e_stats.npz (histograms, pair counts), and per config x variant x
threshold rec_<cfg>_<variant>_<thr>.npz with slot (int64 [K], slot index) and fail (uint8 [K, M, 100], failed
segments of each stream of every slot with any failure); tx_modes gives the LG order of each stream.
usage: python run_e2e.py smoke 20000 14        (first N slots -> results/e2e/smoke/)
       python run_e2e.py run 10 14             (10 parts of 1e6 slots, resumable, then merge)
       python run_e2e.py merge
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(v, "1")
import glob
import json
import math
import pickle
import sys
import time
from multiprocessing import Pool

import numpy as np
from scipy import special, stats

import oamfso as of
import run_modeset as rm
from run_aging import step
from run_sens2 import mmse_filter, sinr_of

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "e2e")
S2 = 2.5; TAU = 2.5; JMAX = 780; J2 = 435
NPT = 11; NSEG = 100; NPIL = 64; XT_DB = -20.0; DISP_RMS = 0.1
CFGS = ("T6R11", "T8R8")
NMAX = {"T6R11": 10_000_000, "T8R8": 1_000_000}
NJIT = int(os.environ.get("E2E_NJIT", 2_000_000))    # slots with integ_jit (it doubles the cost per slot)
VARS = ("paper", "paper435", "integ", "integ_rin165", "integ_rin160", "integ_jit")
THR = {"hd3.8e-3": rm.GTH, "rs1e-3": of.gamma_th_from_ber(1e-3), "rs1.8e-4": of.gamma_th_from_ber(1.8e-4)}
SEED = 20261117                    # chunk i uses SeedSequence(SEED, spawn_key=(i,)); paper runs used 3303, 7300+, 8800000+, 9901
CHUNK = 200
PART = 1_000_000
_ST = None


def init():
    global _ST
    base = of.OAMChannel(j_max=JMAX); gr = base.grid
    U = {m: of.lg_mode(m, gr["r"], gr["phi"], base.w) for m in rm.RX_ALL}
    CR = np.ascontiguousarray((np.conj(np.stack([U[m] for m in rm.RX_ALL])) * gr["dA"]).T)          # (K, 13)
    Q = CR[:, :, None] * np.stack([U[m] for m in rm.TX8]).T[:, None, :]                              # (K, 13, 8)
    sc = of.turbulence(S2)["D_over_r0"] ** (5 / 6)
    n2 = J2 - of.J_MIN + 1
    cx = math.sqrt(10 ** (XT_DB / 10))
    ph = np.random.default_rng(77).uniform(0, 2 * np.pi, size=(len(rm.RX_ALL), 2))                  # sorter phases of run_sens2
    Mx = np.eye(len(rm.RX_ALL), dtype=complex)
    for p in range(len(rm.RX_ALL)):
        if p >= 1:
            Mx[p, p - 1] = cx * np.exp(1j * ph[p, 0])
        if p + 1 < len(rm.RX_ALL):
            Mx[p, p + 1] = cx * np.exp(1j * ph[p, 1])
    Mx /= math.sqrt(1 + 2 * cx * cx)
    rho = math.exp(-(1.0 / TAU) ** (5 / 3))
    _ST = dict(base=base, gr=gr, w=base.w, CR=CR, Q=Q, Mx=Mx, ab=base.gg_params(S2),
               LZ=np.ascontiguousarray((base.Lc.T @ base.Z) * sc),                                     # Phi = z @ LZ
               LZ2=np.ascontiguousarray((base.Lc[:n2, :n2].T @ base.Z[:n2]) * sc), n2=n2,
               rs=rho ** (1 / (NPT - 1)), rp=math.exp(-(0.0005 / TAU) ** (5 / 3)))


def interp_matrix():
    """(NSEG, NPT) weights: SINR_dB at segment midpoints from the NPT points (linear in time)."""
    u = (np.arange(NSEG) + 0.5) / NSEG * (NPT - 1)
    i0 = np.floor(u).astype(int); f = u - i0
    W = np.zeros((NSEG, NPT))
    W[np.arange(NSEG), i0] = 1 - f; W[np.arange(NSEG), i0 + 1] = f
    return W


def irr(ux, uy):
    al, be = _ST["ab"]
    return (special.gammaincinv(al, special.ndtr(ux)) / al)[:, None] * special.gammaincinv(be, special.ndtr(uy)) / be


def expj(Phi):
    E = np.empty(Phi.shape, complex)
    np.cos(Phi, out=E.real); np.sin(Phi, out=E.imag)
    return E


def field(E, ti, Ud=None, nb=8):
    """Complex coupling h (B, 13, len(ti)) from E = exp(j Phi): centred beam (Q) or displaced transmitted modes
    Ud (B, len(ti), K), the latter in blocks of nb slots (kept in cache)."""
    st = _ST
    if Ud is None:
        key = tuple(ti)
        if key not in st:
            st[key] = np.ascontiguousarray(st["Q"][:, :, ti].reshape(E.shape[1], -1))
        return (E @ st[key]).reshape(E.shape[0], len(rm.RX_ALL), len(ti))
    h = np.empty((E.shape[0], len(ti), len(rm.RX_ALL)), complex)
    for b in range(0, E.shape[0], nb):
        h[b:b + nb] = (Ud[b:b + nb] * E[b:b + nb, None, :]) @ st["CR"]
    return h.transpose(0, 2, 1)


def gain(h, I, xt=True):
    if xt:
        h = np.einsum("pq,bqn->bpn", _ST["Mx"], h)
    return np.abs(h) ** 2 * I[:, None, :]


def work(i):
    st = _ST; gr = st["gr"]
    g0 = i * CHUNK
    rng = np.random.default_rng(np.random.SeedSequence(SEED, spawn_key=(i,)))
    n = CHUNK
    cfgs = [k for k in CFGS if g0 < NMAX[k]]
    tx_act = [m for m in rm.TX8 if any(m in rm.CFG[k][0] for k in cfgs)]
    ti = [rm.TX8.index(m) for m in tx_act]
    sel = {k: ([rm.RX_ALL.index(m) for m in rm.CFG[k][1]], [tx_act.index(m) for m in rm.CFG[k][0]]) for k in cfgs}
    jit = g0 < NJIT
    vars_ = [v for v in VARS if jit or v != "integ_jit"]
    z = rng.standard_normal((n, st["LZ"].shape[0])); ux = rng.standard_normal(n); uy = rng.standard_normal((n, len(rm.TX8)))
    d = rng.standard_normal((n, 2)) * (DISP_RMS * st["w"] / math.sqrt(2))
    if jit:
        xs, ys = gr["x"][None, :] - d[:, :1], gr["y"][None, :] - d[:, 1:]
        rs_, ps_ = np.hypot(xs, ys), np.arctan2(ys, xs)
        Ud = np.stack([of.lg_mode(m, rs_, ps_, st["w"]) for m in tx_act], 1)                         # (n, M_act, K)
        del xs, ys, rs_, ps_
    Pk = {k: {"p": rm.snr0(len(rm.CFG[k][0])), "r165": rm.snr0(len(rm.CFG[k][0]), 0.0, -165.0),
              "r160": rm.snr0(len(rm.CFG[k][0]), 0.0, -160.0)} for k in cfgs}
    g = {k: {v: np.empty((n, len(rm.CFG[k][0]), NPT)) for v in vars_} for k in cfgs}
    qp = math.sqrt(1 - st["rp"] ** 2)
    for t in range(NPT):
        if t:
            z, ux, uy = step(rng, z, ux, uy, st["rs"])
        uxp = st["rp"] * ux + qp * rng.standard_normal(ux.shape); uyp = st["rp"] * uy + qp * rng.standard_normal(uy.shape)
        e = rng.standard_normal((n, len(rm.RX_ALL), len(rm.TX8)))
        I = irr(ux, uy)[:, ti]; Ip = irr(uxp, uyp)[:, ti]
        Phi = z @ st["LZ"]
        # pilot channel 0.5 us earlier: Phi_p = rp Phi + qp (w @ LZ); w @ LZ is taken as the screen of slot b+1+t of the
        # chunk at this point (standard normal coefficients, independent of slot b and of the other points of slot b)
        E, Ep = expj(Phi), expj(st["rp"] * Phi + qp * np.roll(Phi, -(1 + t), axis=0))
        del Phi
        h = field(E, ti)
        Gx, Gxp = gain(h, I), gain(field(Ep, ti), Ip)
        if jit:
            Gd, Gdp = gain(field(E, ti, Ud), I), gain(field(Ep, ti, Ud), Ip)
        if t == 0:
            G0 = gain(h, I, False)
            G2 = gain(field(expj(z[:, :st["n2"]] @ st["LZ2"]), ti), I, False)
        del E, Ep
        for k in cfgs:
            ri, ci = sel[k]; P = Pk[k]
            sub = lambda G: G[:, ri][:, :, ci]
            ek = e[:, ri][:, :, [rm.TX8.index(m) for m in rm.CFG[k][0]]]
            if t == 0:
                g[k]["paper"][:] = of.sinr_mmse(sub(G0), P["p"])[:, :, None]
                g[k]["paper435"][:] = of.sinr_mmse(sub(G2), P["p"])[:, :, None]
            A, Ap = sub(Gx), sub(Gxp)
            for v, pk in (("integ", "p"), ("integ_rin165", "r165"), ("integ_rin160", "r160")):
                g[k][v][:, :, t] = sinr_of(mmse_filter(Ap + ek / math.sqrt(NPIL * P[pk]), P[pk]), A, P[pk])[0]
            if jit:
                A, Ap = sub(Gd), sub(Gdp)
                g[k]["integ_jit"][:, :, t] = sinr_of(mmse_filter(Ap + ek / math.sqrt(NPIL * P["p"]), P["p"]), A, P["p"])[0]
    Wi = interp_matrix()
    out = {}
    for k in cfgs:
        M = len(rm.CFG[k][0])
        for v in vars_:
            sdb = (10 * np.log10(np.maximum(g[k][v], 1e-30))) @ Wi.T                                     # (n, M, NSEG)
            for tn, gt in THR.items():
                F = sdb < 10 * math.log10(gt)
                sf = F.any(2); bad = sf.any(1)
                nseg = F.sum(2)
                Fb = F[bad]
                Fs = Fb.transpose(1, 0, 2).reshape(M, -1).astype(np.int64)
                out[(k, v, tn)] = dict(
                    N=n, out=int(bad.sum()), segfail=F.sum((0, 2)).astype(np.int64),
                    nfail=np.bincount(sf.sum(1), minlength=M + 1),
                    nseg=np.bincount(nseg[sf], minlength=NSEG + 1),
                    nsegtot=np.bincount(nseg.sum(1), minlength=M * NSEG + 1),
                    pair=sf.T.astype(np.int64) @ sf.astype(np.int64), pairseg=Fs @ Fs.T,
                    slot=g0 + np.flatnonzero(bad), bits=np.packbits(Fb, axis=-1))
    return out


def add(acc, r):
    for key, d in r.items():
        a = acc.get(key)
        if a is None:
            acc[key] = {f: (x.copy() if f not in ("slot", "bits") else [x]) if isinstance(x, np.ndarray) else x for f, x in d.items()}
            continue
        for f, x in d.items():
            if f in ("slot", "bits"):
                a[f].append(x)
            else:
                a[f] = a[f] + x


def run_range(lo, hi, workers, tag=""):
    t0 = time.time()
    jobs = list(range(lo // CHUNK, math.ceil(hi / CHUNK)))
    acc = {}
    with Pool(workers, initializer=init) as pool:
        for j, r in enumerate(pool.imap_unordered(work, jobs, chunksize=1)):
            add(acc, r)
            if (j + 1) % 250 == 0:
                print(f"{tag} {j + 1}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    for a in acc.values():
        a["slot"] = np.concatenate(a["slot"]); a["bits"] = np.concatenate(a["bits"])
    print(f"{tag} slots {lo}-{hi} in {time.time() - t0:.0f}s", flush=True)
    return acc, time.time() - t0


def cp(k, n, a=0.05):
    lo = 0.0 if k == 0 else float(stats.beta.ppf(a / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(stats.beta.ppf(1 - a / 2, k + 1, n - k))
    return [lo, hi]


def save(parts, outdir, runtime):
    os.makedirs(outdir, exist_ok=True)
    acc = {}
    for p in parts:
        for key, d in p.items():
            add(acc, {key: d})
    res = {"s2": S2, "tau_c_ms": TAU, "J": JMAX, "J_alt": J2, "slot_ms": 1.0, "points_per_slot": NPT, "point_spacing_us": 100.0,
           "segments_per_slot": NSEG, "segment_us": 10.0, "bits_per_segment_per_stream": 100_000, "baud": 10e9,
           "pilot_symbols": NPIL, "pilot_period_us": 1.0, "xt_dB": XT_DB, "disp_rms_over_w": DISP_RMS, "w_L_m": beam_w(),
           "thresholds_dB": {t: 10 * math.log10(x) for t, x in THR.items()}, "seed": SEED, "chunk": CHUNK,
           "runtime_s": runtime, "configs": {}}
    arrs = {}
    for (k, v, tn), a in sorted(acc.items()):
        a["slot"] = np.concatenate(a["slot"]); a["bits"] = np.concatenate(a["bits"])
        o = np.argsort(a["slot"], kind="stable")
        N = a["N"]; M = len(rm.CFG[k][0]); ns = a["nseg"]
        res["configs"].setdefault(k, {"tx_modes": list(rm.CFG[k][0]), "rx_modes": list(rm.CFG[k][1]), "N": N, "variants": {}})
        res["configs"][k]["variants"].setdefault(v, {})[tn] = {
            "N": N, "outage": a["out"] / N, "outage_count": a["out"], "outage_cp95": cp(a["out"], N),
            "seg_err_rate": (a["segfail"] / (N * NSEG)).tolist(),
            "seg_err_rate_mean": float(a["segfail"].sum() / (N * NSEG * M)),
            "stream_slot_fail_rate": (np.diagonal(a["pair"]) / N).tolist(),
            "mean_failed_fraction_of_failed_stream": float((np.arange(NSEG + 1) * ns).sum() / max(ns[1:].sum(), 1) / NSEG),
            "nfail_hist": a["nfail"].tolist()}
        for f in ("nfail", "nseg", "nsegtot", "pair", "pairseg", "segfail"):
            arrs[f"{k}/{v}/{tn}/{f}"] = a[f]
        fail = np.unpackbits(a["bits"][o], axis=-1, count=NSEG).astype(np.uint8)
        np.savez_compressed(os.path.join(outdir, f"rec_{k}_{v}_{tn}.npz"), slot=a["slot"][o].astype(np.int64), fail=fail,
                            tx_modes=np.array(rm.CFG[k][0]), rx_modes=np.array(rm.CFG[k][1]), N=N)
        print(k, v, tn, f"out {a['out'] / N:.4g} {cp(a['out'], N)} seg {a['segfail'].sum() / (N * NSEG * M):.3g} "
              f"ffrac {res['configs'][k]['variants'][v][tn]['mean_failed_fraction_of_failed_stream']:.3f}", flush=True)
    np.savez_compressed(os.path.join(outdir, "e2e_stats.npz"), **arrs)
    json.dump(res, open(os.path.join(outdir, "e2e_summary.json"), "w"), indent=1)


def beam_w():
    return of.beam_params()[2]


def main(mode, args):
    if mode == "smoke":
        N, workers = int(args[0]), int(args[1])
        acc, rt = run_range(0, N, workers, "smoke")
        save([acc], os.path.join(OUT, "smoke"), rt)
    elif mode in ("run", "merge"):
        pdir = os.path.join(OUT, "parts"); os.makedirs(pdir, exist_ok=True)
        if mode == "run":
            nparts, workers = int(args[0]), int(args[1])
            for p in range(nparts):
                fn = os.path.join(pdir, f"part_{p:03d}.pkl")
                if os.path.exists(fn):
                    continue
                acc, rt = run_range(p * PART, (p + 1) * PART, workers, f"part{p}")
                pickle.dump((acc, rt), open(fn + ".tmp", "wb"), protocol=4); os.replace(fn + ".tmp", fn)
        fns = sorted(glob.glob(os.path.join(pdir, "part_*.pkl")))
        parts, rts = zip(*[pickle.load(open(f, "rb")) for f in fns])
        save(parts, OUT, float(sum(rts)))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
