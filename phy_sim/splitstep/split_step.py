"""Multi-screen split-step (angular-spectrum) wave-optics reference for the OAM-FSO channel of oamfso.py.

Every transmitted LG_{0,l} mode of TX8 (and a Gaussian LG_{0,0} beacon with the same waist) is launched
at z = 0 with the waist of oamfso.beam_params() and propagated over L = 1500 m through the SAME set of
Ns Kolmogorov phase screens per realization (Fourier-series screens with three levels of subharmonics,
Lane et al. 1992 / Johansson and Gavel 1994, Schmidt 2010 listing 9.3). Turbulence: constant Cn2 equal
to oamfso.turbulence(sigmaR2)["Cn2"], Kolmogorov spectrum (L0 = infinity, l0 = 0; the grid truncates
the spectrum at 1/(2 dx)). The screens sit at the midpoints of Ns equal slabs; their strengths
w_i = Cn2_i dz_i are the minimum-norm correction of the uniform split that reproduces exactly the
plane-wave r0 and the plane-wave Rytov variance of the continuous path.

Propagation: exact angular-spectrum transfer function exp(i dz (k_z - k)), super-Gaussian absorbing
boundary exp(-(r / (0.47 N dx))^16) after every partial propagation (Schmidt 2010, ch. 9).

Receiver: 0.20 m aperture. The overlap h_rn = sum_aperture conj(u_r) U_n exp(-i k r^2 / (2 R(L))) dx^2,
u_r = oamfso.lg_mode(r, ., ., w(L)) (unit power on the full plane, as in oamfso), r = -6..6; the
exp(-i k r^2/2R) factor removes the vacuum wavefront curvature for which the mode sorter is designed
(vacuum check: h_nn equals the oamfso vacuum value). G = |h|^2 contains scintillation, beam spreading,
beam wander and crosstalk jointly (no separate Gamma-Gamma factor). Three receiver variants:
  raw : no tracking;
  tt  : receive-side tip/tilt correction, G-tilt of the beacon over the aperture (fast steering mirror,
        the counterpart of the removal of Zernike j = 2, 3 in oamfso);
  ttc : tt plus recentring of the sorter modes on the beacon centroid in the aperture (idealized
        upper bound; beam wander at the receive plane cannot be removed by a receive-side mirror).

Subcommands: validate | run | zern | analyze (see README.md).
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import math
import multiprocessing as mp
import sys
import time

import numpy as np
from scipy import fft as sfft
from scipy import stats as sst

HERE = os.path.dirname(os.path.abspath(__file__))
SIM = os.path.dirname(HERE)                     # phy_sim/ (oamfso.py, run_modeset.py)
if not os.path.exists(os.path.join(SIM, "oamfso.py")):
    SIM = os.path.join(os.path.dirname(HERE), "sim")   # layout of the original run directory
sys.path.insert(0, SIM)
import oamfso as of            # noqa: E402
import run_modeset as rm       # noqa: E402

LAM, K, L = of.LAMBDA, of.K_WAVE, of.L_LINK
A = of.A_R
W0, ZR, WL = of.beam_params()
RL = L * (1.0 + (ZR / L) ** 2)                 # vacuum wavefront radius at the receiver
TX8, RX_ALL = rm.TX8, rm.RX_ALL
CFGS = ("T8R8", "T6R8", "T6R11")
GTH = rm.GTH
N_DEF, DX_DEF, NS_DEF = 512, 2e-3, 10
SEEDS = {  # split-step main runs, Ns = 20 check, grid checks, Zernike matched / large, validation
    "ss": {1.0: 6101, 2.5: 6102}, "ss20": {1.0: 6201, 2.5: 6202}, "gridfine": {1.0: 6301, 2.5: 6302},
    "gridwide": {1.0: 6311, 2.5: 6312}, "ss6sh": {1.0: 6401, 2.5: 6402},
    "zern": {1.0: 6501, 2.5: 6502}, "zernbig": {1.0: 6601, 2.5: 6602},
    "val_sf": 6001, "val_pw": 6002, "val_gb": 6003, "vac": 0}
CHUNK = 20


# ----------------------------------------------------------------------------- turbulence layout
def screen_layout(s2, Ns):
    """Screen positions (midpoints of Ns equal slabs) and strengths w_i = Cn2_i dz_i. The uniform split
    w_i = Cn2 L / Ns is corrected by the minimum-norm vector that makes the discrete plane-wave r0
    (sum w_i) and plane-wave Rytov variance (2.25 k^(7/6) sum w_i (L - z_i)^(5/6)) equal to the
    continuous values for constant Cn2."""
    Cn2 = of.turbulence(s2)["Cn2"]
    z = (np.arange(Ns) + 0.5) * L / Ns
    w = np.full(Ns, Cn2 * L / Ns)
    Amat = np.vstack([np.ones(Ns), (L - z) ** (5 / 6)])
    b = np.array([Cn2 * L, (6 / 11) * Cn2 * L ** (11 / 6)])
    w = w + Amat.T @ np.linalg.solve(Amat @ Amat.T, b - Amat @ w)
    if (w <= 0).any():
        raise ValueError("negative screen strength")
    return z, w


def layout_summary(s2, Ns):
    z, w = screen_layout(s2, Ns)
    t = of.turbulence(s2)
    Cn2 = t["Cn2"]
    r0_i = (0.423 * K ** 2 * w) ** (-3 / 5)
    r0_pw = (0.423 * K ** 2 * w.sum()) ** (-3 / 5)
    r0_sw = (0.423 * K ** 2 * np.sum(w * (z / L) ** (5 / 3))) ** (-3 / 5)
    rytov_pw = 2.25 * K ** (7 / 6) * np.sum(w * (L - z) ** (5 / 6))
    rytov_sw = 2.25 * K ** (7 / 6) * np.sum(w * (z / L) ** (5 / 6) * (L - z) ** (5 / 6))
    dz = L / Ns
    return {"Ns": Ns, "z_m": z.tolist(), "Cn2dz": w.tolist(), "Cn2_eff_per_screen": (w / dz).tolist(), "r0_screen_m": r0_i.tolist(),
            "Cn2_target": Cn2, "r0_pw_target_m": t["r0"], "r0_pw_screens_m": float(r0_pw),
            "rytov_pw_target": s2, "rytov_pw_screens": float(rytov_pw),
            "r0_sw_continuous_m": float((0.423 * K ** 2 * Cn2 * L * 3 / 8) ** (-3 / 5)), "r0_sw_screens_m": float(r0_sw),
            "rytov_sw_continuous": float(2.25 * K ** (7 / 6) * Cn2 * L ** (11 / 6) * math.gamma(11 / 6) ** 2 / math.gamma(11 / 3)), "rytov_sw_screens": float(rytov_sw),
            "rytov_pw_partial_step": float(1.23 * Cn2 * K ** (7 / 6) * dz ** (11 / 6))}


# ----------------------------------------------------------------------------- grid and propagation
class Grid:
    def __init__(self, N, dx, cdtype=np.complex64, nsub=3):
        self.N, self.dx, self.cdtype, self.nsub = N, dx, cdtype, nsub
        x = (np.arange(N) - N // 2) * dx
        self.x = x
        self.X, self.Y = np.meshgrid(x, x, indexing="xy")
        self.R = np.hypot(self.X, self.Y)
        self.PHI = np.arctan2(self.Y, self.X)
        f = sfft.fftfreq(N, dx)
        FX, FY = np.meshgrid(f, f, indexing="xy")
        self.F2 = FX ** 2 + FY ** 2
        self.sg = np.exp(-(self.R / (0.47 * N * dx)) ** 16).astype(np.float32 if cdtype == np.complex64 else np.float64)
        with np.errstate(divide="ignore"):
            psd = 0.023 * self.F2 ** (-11 / 6)
        psd[0, 0] = 0.0
        self.sqrt_psd_df = np.sqrt(psd) / (N * dx)            # r0 = 1 m
        self._H = {}

    def H(self, dz):
        key = round(float(dz), 9)
        if key not in self._H:
            kk = 4 * math.pi ** 2 * self.F2
            self._H[key] = np.exp(1j * dz * (-kk / (K + np.sqrt(K ** 2 - kk)))).astype(self.cdtype)  # exp(i dz (k_z - k))
        return self._H[key]

    def prop(self, U, dz):
        """Angular-spectrum step of length dz followed by the absorbing boundary (U is overwritten)."""
        F = sfft.fft2(U, axes=(-2, -1), overwrite_x=True)
        F *= self.H(dz)
        U = sfft.ifft2(F, axes=(-2, -1), overwrite_x=True)
        U *= self.sg
        return U

    def screen_pair(self, rng):
        """Two independent Kolmogorov phase screens for r0 = 1 m (scale by r0^(-5/6)); real and imaginary
        parts of one complex Fourier-series screen plus nsub (default three) levels of 3 x 3 subharmonics."""
        N, dx = self.N, self.dx
        cn = (rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))) * self.sqrt_psd_df
        scr = sfft.ifft2(cn, overwrite_x=True) * (N * N)
        lo = np.zeros((N, N), complex)
        df = 1.0 / (N * dx)
        for p in range(1, self.nsub + 1):
            dfp = df / 3 ** p
            fx = np.array([-1.0, 0.0, 1.0]) * dfp
            FXs, FYs = np.meshgrid(fx, fx, indexing="xy")
            with np.errstate(divide="ignore"):
                psd = 0.023 * (FXs ** 2 + FYs ** 2) ** (-11 / 6)
            psd[1, 1] = 0.0
            c = (rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3))) * np.sqrt(psd) * dfp
            ex = np.exp(2j * math.pi * fx[:, None] * self.x[None, :])      # (3, N): index b -> fx_b x
            lo += ex.T @ c @ ex                                            # lo[y, x] = sum_ab c_ab e^{i2pi(fy_a y + fx_b x)}
        lo -= lo.mean()
        scr += lo
        return scr.real, scr.imag


def lg_field(m, X, Y, w, xc=0.0, yc=0.0):
    r = np.hypot(X - xc, Y - yc)
    return of.lg_mode(m, r, np.arctan2(Y - yc, X - xc), w)


def sampling_checks(s2_list, N, dx, Ns, c_list=(2.0, 4.0)):
    """Schmidt (2010) eqs. 9.86-9.89 / 7.x: constraints for equal source and observation spacing."""
    u = sst.gamma.ppf(1 - 1e-4, max(abs(m) for m in TX8) + 1)       # 99.99 % power of LG_{0,4} at z = 0
    D1 = 2 * W0 * math.sqrt(u / 2)
    D2 = of.D_R
    out = {"N": N, "dx_m": dx, "grid_side_m": N * dx, "absorber": "exp(-(r/(0.47 N dx))^16)", "D1_source_m": D1,
           "D1_definition": "diameter with 99.99 % of the LG_{0,4} power at z = 0", "D2_receiver_m": D2,
           "dz_max_m": float(np.max(np.diff([0, *screen_layout(1.0, Ns)[0], L]))), "levels": {}}
    out["c3_partial_dz_limit_m"] = dx ** 2 * N / LAM
    out["c3_ok"] = out["dz_max_m"] <= out["c3_partial_dz_limit_m"]
    for s2 in s2_list:
        Cn2 = of.turbulence(s2)["Cn2"]
        r0sw = (0.423 * K ** 2 * Cn2 * L * 3 / 8) ** (-3 / 5)
        lv = {"r0_sw_m": r0sw, "r0_pw_m": of.turbulence(s2)["r0"],
              "r0_screen_min_over_dx": float(min((0.423 * K ** 2 * screen_layout(s2, Ns)[1]) ** (-3 / 5)) / dx),
              "r0_pw_over_dx": of.turbulence(s2)["r0"] / dx}
        for c in c_list:
            D1p = D1 + c * LAM * L / r0sw
            D2p = D2 + c * LAM * L / r0sw
            dmax = LAM * L / (D1p + D2p)
            Nmin = D1p / (2 * dx) + D2p / (2 * dx) + LAM * L / (2 * dx * dx)
            lv[f"c={c:g}"] = {"D1p_m": D1p, "D2p_m": D2p, "c1_dx_max_m": dmax, "c1_ok": dx <= dmax,
                              "c2_N_min": Nmin, "c2_ok": N >= Nmin}
        out["levels"][str(s2)] = lv
    return out


# ----------------------------------------------------------------------------- link simulation
class Link:
    def __init__(self, s2, N, dx, Ns, cdtype=np.complex64, nsub=3):
        self.g = g = Grid(N, dx, cdtype, nsub)
        self.s2, self.Ns = s2, Ns
        self.z, self.w = screen_layout(s2, Ns)
        self.r0 = (0.423 * K ** 2 * self.w) ** (-3 / 5)
        self.dzs = np.diff([0.0, *self.z, L])
        self.U0 = np.stack([lg_field(m, g.X, g.Y, W0) for m in TX8] + [lg_field(0, g.X, g.Y, W0)]).astype(cdtype)
        # crop around the aperture
        n = int(math.ceil(A / dx)) + 2
        c0 = N // 2
        self.sl = slice(c0 - n, c0 + n + 1)
        Xc, Yc = g.X[self.sl, self.sl], g.Y[self.sl, self.sl]
        self.Xc, self.Yc = Xc, Yc
        self.ap = (np.hypot(Xc, Yc) <= A).astype(float)
        self.curv = np.exp(-1j * K * (Xc ** 2 + Yc ** 2) / (2 * RL))
        self.Ur = np.stack([lg_field(m, Xc, Yc, WL) for m in RX_ALL]) * self.ap          # (13, nc, nc)

    def overlaps(self, Uc, Ur):
        """h[r, n] = sum conj(u_r) U_n dx^2 over the aperture (Ur already masked)."""
        return np.einsum("rij,nij->rn", np.conj(Ur), Uc) * self.g.dx ** 2

    def propagate(self, rng, screens=True):
        """All 9 fields through the same Ns screens. The fields are propagated one at a time (a 2 MB
        working set per field keeps the FFTs in the shared L3 cache when many workers run)."""
        g = self.g
        if screens:
            pairs = [g.screen_pair(rng) for _ in range((self.Ns + 1) // 2)]
            E = [np.exp(1j * (pairs[i // 2][i % 2] * self.r0[i] ** (-5 / 6))).astype(g.cdtype) for i in range(self.Ns)]
            del pairs
        out = np.empty_like(self.U0)
        for f in range(self.U0.shape[0]):
            U = g.prop(self.U0[f].copy(), self.dzs[0])
            for i in range(self.Ns):
                if screens:
                    U *= E[i]
                U = g.prop(U, self.dzs[i + 1])
            out[f] = U
        return out

    def receive(self, U):
        g, dx2 = self.g, self.g.dx ** 2
        Pgrid = (np.abs(U).astype(np.float64) ** 2).sum((-2, -1)) * dx2
        B = U[-1].astype(np.complex128)
        IB = np.abs(B) ** 2
        cen_full = np.array([(g.X * IB).sum(), (g.Y * IB).sum()]) / IB.sum()
        Uc = U[:, self.sl, self.sl].astype(np.complex128) * self.curv
        Pap = (np.abs(Uc) ** 2 * self.ap).sum((-2, -1)) * dx2
        b = Uc[-1]
        wb = np.abs(b) ** 2 * self.ap
        gy, gx = np.gradient(b, g.dx)
        tilt = np.array([(np.imag(np.conj(b) * gx) * self.ap).sum(), (np.imag(np.conj(b) * gy) * self.ap).sum()]) / (K * wb.sum())
        cen = np.array([(self.Xc * wb).sum(), (self.Yc * wb).sum()]) / wb.sum()
        Um = Uc[:8]
        h_raw = self.overlaps(Um, self.Ur)
        Ut = Um * np.exp(-1j * K * (tilt[0] * self.Xc + tilt[1] * self.Yc))
        h_tt = self.overlaps(Ut, self.Ur)
        Urc = np.stack([lg_field(m, self.Xc, self.Yc, WL, cen[0], cen[1]) for m in RX_ALL]) * self.ap
        h_ttc = self.overlaps(Ut, Urc)
        return {"G_raw": np.abs(h_raw) ** 2, "G_tt": np.abs(h_tt) ** 2, "G_ttc": np.abs(h_ttc) ** 2,
                "Pap": Pap, "Pgrid": Pgrid, "tilt": tilt, "cen_ap": cen, "cen_full": cen_full}


_LINKS = {}


def _get_link(s2, N, dx, Ns, nsub=3):
    key = (s2, N, dx, Ns, nsub)
    if key not in _LINKS:
        _LINKS.clear()
        _LINKS[key] = Link(s2, N, dx, Ns, nsub=nsub)
    return _LINKS[key]


def _chunk(task):
    s2, N, dx, Ns, ss, n = task[:6]
    link = _get_link(s2, N, dx, Ns, task[6] if len(task) > 6 else 3)
    rng = np.random.default_rng(ss)
    rows = [link.receive(link.propagate(rng)) for _ in range(n)]
    return {k: np.stack([r[k] for r in rows]) for k in rows[0]}


def run_splitstep(s2, Nreal, N, dx, Ns, seed, workers, tag, nsub=3):
    t0 = time.time()
    ntask = int(math.ceil(Nreal / CHUNK))
    kids = np.random.SeedSequence(seed).spawn(ntask)
    tasks = [(s2, N, dx, Ns, kids[i], min(CHUNK, Nreal - i * CHUNK), nsub) for i in range(ntask)]
    res = [None] * ntask
    ctx = mp.get_context("forkserver")
    with ctx.Pool(workers) as pool:
        for i, r in enumerate(pool.imap(_chunk, tasks)):
            res[i] = r
            if (i + 1) % max(1, ntask // 20) == 0 or i + 1 == ntask:
                print(f"{tag} s2={s2}: {min((i + 1) * CHUNK, Nreal)}/{Nreal} realizations, {time.time() - t0:.0f} s", flush=True)
    out = {k: np.concatenate([r[k] for r in res]) for k in res[0]}
    meta = {"s2": s2, "Nreal": Nreal, "N": N, "dx": dx, "Ns": Ns, "nsub": nsub, "seed": seed, "workers": workers, "chunk": CHUNK,
            "runtime_s": time.time() - t0, "tag": tag}
    np.savez_compressed(os.path.join(HERE, f"raw_{tag}_{s2}.npz"), meta=json.dumps(meta), **out)
    print(f"{tag} s2={s2} done in {meta['runtime_s']:.0f} s", flush=True)
    return meta


# ----------------------------------------------------------------------------- Zernike reference model
def run_zernike(s2, Nreal, seed, tag, batch=2000):
    t0 = time.time()
    ch = of.OAMChannel(modes=TX8, rx_modes=RX_ALL)
    vac_ap = np.array([np.sum(np.abs(of.lg_mode(m, ch.grid["r"], ch.grid["phi"], ch.w)) ** 2) * ch.grid["dA"] for m in TX8])
    rng = np.random.default_rng(seed)
    G = np.empty((Nreal, len(RX_ALL), len(TX8)), np.float64 if Nreal <= 200000 else np.float32)
    I = np.empty((Nreal, len(TX8)))
    done = 0
    while done < Nreal:
        b = min(batch, Nreal - done)
        Gb, _, Ib = ch.draw(rng, b, s2, return_parts=True)
        G[done:done + b] = Gb; I[done:done + b] = Ib
        done += b
    meta = {"s2": s2, "Nreal": Nreal, "seed": seed, "J_max": of.J_MAX, "gg_alpha_beta": ch.gg_params(s2),
            "runtime_s": time.time() - t0, "tag": tag, "vac_ap": vac_ap.tolist()}
    np.savez_compressed(os.path.join(HERE, f"raw_{tag}_{s2}.npz"), meta=json.dumps(meta), G=G, I=I, Pap=I * vac_ap[None, :])
    print(f"{tag} s2={s2} done in {meta['runtime_s']:.0f} s", flush=True)


# ----------------------------------------------------------------------------- statistics
def wilson(k, n, z=1.959963984540054):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [0.0 if k == 0 else max(0.0, c - h), 1.0 if k == n else min(1.0, c + h)]


def stats_G(G, Pap, chunk=100000):
    G = np.asarray(G)
    n = G.shape[0]
    Gm = np.zeros(G.shape[1:])
    for i in range(0, n, chunk):
        Gm += G[i:i + chunk].astype(np.float64).sum(0)
    Gm /= n
    ti = {m: TX8.index(m) for m in TX8}
    ri = {m: RX_ALL.index(m) for m in RX_ALL}
    self_c = np.array([Gm[ri[m], ti[m]] for m in TX8])
    xt8 = np.array([sum(Gm[ri[r], ti[m]] for r in TX8 if r != m) for m in TX8])
    xt13 = np.array([sum(Gm[ri[r], ti[m]] for r in RX_ALL if r != m) for m in TX8])
    db = lambda v: (10 * np.log10(v)).tolist()
    out = {"N": n, "mean_G_13x8": Gm.tolist(), "rx_modes": list(RX_ALL), "tx_modes": list(TX8),
           "self_dB": db(self_c), "self_dB_mean_over_modes": float(10 * np.log10(self_c.mean())),
           "xt_TX8ports_dB": db(xt8), "xt_TX8ports_dB_mean": float(10 * np.log10(xt8.mean())),
           "xt_13ports_dB": db(xt13), "xt_13ports_dB_mean": float(10 * np.log10(xt13.mean())),
           "xt_over_self_TX8ports_dB": db(xt8 / self_c), "xt_over_self_TX8ports_dB_mean": float(10 * np.log10(np.mean(xt8 / self_c)))}
    # received power per mode: aperture power and the sum over the 13 sorter ports
    P13 = np.concatenate([G[i:i + chunk].astype(np.float64).sum(1) for i in range(0, n, chunk)])
    qs = (1e-3, 1e-2, 1e-1, 0.5)
    pw = {}
    for name, P in (("aperture", np.asarray(Pap)[:, :8]), ("ports13", P13)):
        mu = P.mean(0)
        si = P.var(0) / mu ** 2
        pw[name] = {"mean_dB": db(mu), "scint_index": si.tolist(),
                    "quantiles_dB": {str(q): db(np.quantile(P, q, axis=0)) for q in qs},
                    "pooled_quantiles_rel_mean_dB": {str(q): float(10 * np.log10(np.quantile((P / mu).ravel(), q))) for q in qs},
                    "hist_rel_mean_dB": {"edges": list(range(-40, 11)), "counts": np.histogram(10 * np.log10(np.maximum(P / mu, 1e-30)),
                                                                                          bins=np.arange(-40, 11))[0].tolist()}}
    out["power"] = pw
    out["self_scint_index"] = [float(np.var(G[:, ri[m], ti[m]].astype(float)) / self_c[ti[m]] ** 2) for m in TX8]
    # MMSE outage at the link-budget operating point and on the reference-SNR grid
    cf = {}
    for k in CFGS:
        tx, rx = rm.CFG[k]
        M = len(tx)
        snrs = [10 ** (d / 10) for d in rm.SNR_GRID_DB] + [rm.snr0(M)]
        out_cnt = np.zeros(len(snrs), np.int64)
        nfail = np.zeros(M + 1, np.int64)
        pm = np.zeros(M, np.int64)
        ridx = [RX_ALL.index(m) for m in rx]; tidx = [TX8.index(m) for m in tx]
        for i in range(0, n, chunk):
            Gs = G[i:i + chunk][:, ridx][:, :, tidx].astype(np.float64)
            lam, V2 = of.gram_eig(Gs)
            for j, snr in enumerate(snrs):
                fail = of.sinr_mmse_eig(lam, V2, snr) < GTH
                out_cnt[j] += int(fail.any(1).sum())
                if j == len(snrs) - 1:
                    nfail += np.bincount(fail.sum(1), minlength=M + 1)
                    pm += fail.sum(0)
        k_op = int(out_cnt[-1])
        cf[k] = {"tx": list(tx), "rx": list(rx), "snr_op_dB": 10 * math.log10(snrs[-1]), "outage_op": k_op / n, "count_op": k_op,
                 "outage_op_ci95_wilson": wilson(k_op, n), "nfail_hist": nfail.tolist(), "per_mode_outage": (pm / n).tolist(),
                 "snr_grid_dB": list(rm.SNR_GRID_DB), "outage_grid": (out_cnt[:-1] / n).tolist()}
    out["configs"] = cf
    return out


def analyze(s2):
    res = {"s2": s2, "turbulence": of.turbulence(s2), "gamma_th_dB": 10 * math.log10(GTH),
           "beam": {"w0_m": W0, "zR_m": ZR, "wL_m": WL, "R_L_m": RL}, "models": {}}
    vpath = os.path.join(HERE, "validation.json")
    if os.path.exists(vpath):
        v = json.load(open(vpath))
        res["sampling"] = v.get("sampling")
        res["layout"] = v.get("layout", {}).get(str(s2))
    for tag in ("ss", "ss20", "ss6sh", "gridfine", "gridwide", "zern", "zernbig"):
        p = os.path.join(HERE, f"raw_{tag}_{s2}.npz")
        if not os.path.exists(p):
            continue
        d = np.load(p)
        meta = json.loads(str(d["meta"]))
        if tag.startswith("zern"):
            res["models"][tag] = {"meta": meta, "stats": stats_G(d["G"], d["Pap"])}
        else:
            ent = {"meta": meta}
            for var in ("raw", "tt", "ttc"):
                ent[var] = stats_G(d[f"G_{var}"], d["Pap"])
            cf, ca = d["cen_full"], d["cen_ap"]
            ent["beacon"] = {"wander_rms_full_m": float(np.sqrt(np.mean(np.sum(cf ** 2, 1)))),
                             "wander_rms_aperture_m": float(np.sqrt(np.mean(np.sum(ca ** 2, 1)))),
                             "wander_theory_rms_m": float(math.sqrt(2.42 * of.turbulence(s2)["Cn2"] * L ** 3 * W0 ** (-1 / 3))),
                             "tilt_rms_rad": float(np.sqrt(np.mean(np.sum(d["tilt"] ** 2, 1))))}
            ent["power_left_on_grid_min"] = float(d["Pgrid"].min())
            ent["power_left_on_grid_mean"] = np.mean(d["Pgrid"], 0).tolist()
            res["models"][tag] = ent
    with open(os.path.join(HERE, f"results_{s2}.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    write_summary(res)
    return res


def write_summary(res):
    s2 = res["s2"]
    L_ = [f"sigma_R^2 = {s2}: Cn2 = {res['turbulence']['Cn2']:.3e} m^-2/3, r0 = {res['turbulence']['r0'] * 100:.2f} cm, "
          f"D/r0 = {res['turbulence']['D_over_r0']:.2f}, gamma_th = {res['gamma_th_dB']:.2f} dB"]

    def row(name, st):
        s = [f"{name:<22s} N={st['N']:>7d}  self {st['self_dB_mean_over_modes']:6.2f} dB  "
             f"xt(TX8 ports) {st['xt_TX8ports_dB_mean']:6.2f} dB  xt(13) {st['xt_13ports_dB_mean']:6.2f} dB  "
             f"Pap SI {np.mean(st['power']['aperture']['scint_index']):.3f}  P13 mean {np.mean(st['power']['ports13']['mean_dB']):6.2f} dB "
             f"SI {np.mean(st['power']['ports13']['scint_index']):.3f}  self SI {np.mean(st['self_scint_index']):.3f}"]
        for k in CFGS:
            c = st["configs"][k]
            lo, hi = c["outage_op_ci95_wilson"]
            s.append(f"{'':<22s} {k}: P_out = {c['outage_op']:.4g} [{lo:.3g}, {hi:.3g}]  K hist {c['nfail_hist']}")
        return s
    for tag, ent in res["models"].items():
        if tag.startswith("zern"):
            L_ += row(f"Zernike/{tag}", ent["stats"])
        else:
            for var in ("tt", "raw", "ttc"):
                L_ += row(f"split/{tag}/{var}", ent[var])
            L_.append(f"{'':<22s} beacon wander rms {ent['beacon']['wander_rms_full_m'] * 100:.2f} cm "
                      f"(theory {ent['beacon']['wander_theory_rms_m'] * 100:.2f} cm), tilt rms {ent['beacon']['tilt_rms_rad'] * 1e6:.1f} urad, "
                      f"runtime {ent['meta']['runtime_s']:.0f} s")
    with open(os.path.join(HERE, f"summary_{s2}.txt"), "w") as f:
        f.write("\n".join(L_) + "\n")
    print("\n".join(L_))


# ----------------------------------------------------------------------------- validation
def _sf_task(args):
    seed, N, dx, n = args
    g = Grid(N, dx)
    rng = np.random.default_rng(seed)
    offs = np.array([1, 2, 3, 5, 8, 13, 20, 32, 50, 64, 100, 128, 160, 200, 256])
    offs = offs[offs < N]
    acc = np.zeros(offs.size)
    for _ in range(n):
        for s in g.screen_pair(rng):
            acc += np.array([0.5 * (np.mean((s[:, o:] - s[:, :-o]) ** 2) + np.mean((s[o:, :] - s[:-o, :]) ** 2)) for o in offs])
    return offs, acc, 2 * n


def _pw_task(args):
    """Plane-wave scintillation (or Gaussian-beam on-axis scintillation with beam=True)."""
    s2, N, dx, Ns, ss, n, beam = args
    g = Grid(N, dx)
    z, w = screen_layout(s2, Ns)
    r0 = (0.423 * K ** 2 * w) ** (-3 / 5)
    dzs = np.diff([0.0, *z, L])
    rng = np.random.default_rng(ss)
    U0 = lg_field(0, g.X, g.Y, W0) if beam else np.ones((N, N), complex)
    c0 = N // 2
    h = int(round(0.1 / dx))
    box = (slice(c0 - h, c0 + h + 1), slice(c0 - h, c0 + h + 1))

    def go(screens):
        U = g.prop(U0.copy(), dzs[0])
        pairs = [g.screen_pair(rng) for _ in range((Ns + 1) // 2)] if screens else None
        for i in range(Ns):
            if screens:
                U *= np.exp(1j * pairs[i // 2][i % 2] * r0[i] ** (-5 / 6))
            U = g.prop(U, dzs[i + 1])
        return np.abs(U) ** 2
    Ivac = go(False)
    if beam:
        vals = np.array([go(True)[c0, c0] / Ivac[c0, c0] for _ in range(n)])
        return vals, None
    vals = np.stack([go(True)[box] for _ in range(n)])
    return vals, Ivac[box]


def sf_generator_expectation(N, dx, offs, nsub=3):
    """Expected structure function of Grid.screen_pair (discrete spectrum plus the 3 x 3 subharmonic sets),
    D(r) = 2 sum_f PSD(f) df^2 (1 - cos(2 pi f_x r)), r0 = 1 m."""
    g = Grid(N, dx)
    df = 1.0 / (N * dx)
    psd = (g.sqrt_psd_df * (N * dx)) ** 2
    f = sfft.fftfreq(N, dx)
    FX = np.broadcast_to(f[None, :], (N, N))
    out = []
    for o in offs:
        r = o * dx
        D = np.sum(2 * psd * df ** 2 * (1 - np.cos(2 * math.pi * FX * r)))
        for p in range(1, nsub + 1):
            dfp = df / 3 ** p
            fx = np.array([-1.0, 0.0, 1.0]) * dfp
            a_, b_ = np.meshgrid(fx, fx, indexing="xy")
            with np.errstate(divide="ignore"):
                ps = 0.023 * (a_ ** 2 + b_ ** 2) ** (-11 / 6)
            ps[1, 1] = 0.0
            D += np.sum(2 * ps * dfp ** 2 * (1 - np.cos(2 * math.pi * a_ * r)))
        out.append(D)
    return np.array(out)


def _zern_task(args):
    """Noll-Zernike decomposition (j = 1..66, n <= 10, least squares on the grid pixels) of single screens
    and of the sum of the Ns path screens, over the centred D_R = 0.20 m aperture."""
    seed, N, dx, n, s2, Ns = args
    g = Grid(N, dx)
    rng = np.random.default_rng(seed)
    m = g.R <= A
    rho, phi = g.R[m] / A, g.PHI[m]
    Z = np.stack([np.ones_like(rho)] + [of.zernike_noll(j, rho, phi) for j in range(2, 67)], 1)
    pinv = np.linalg.pinv(Z)
    z, w = screen_layout(s2, Ns)
    r0 = (0.423 * K ** 2 * w) ** (-3 / 5)
    a_single, res_single, a_path, res_path = [], [], [], []
    for _ in range(n):
        pairs = [g.screen_pair(rng) for _ in range((Ns + 1) // 2)]
        scr = [pairs[i // 2][i % 2] for i in range(Ns)]
        for ph, store_a, store_r in ((scr[0][m], a_single, res_single),
                                     (sum(scr[i][m] * r0[i] ** (-5 / 6) for i in range(Ns)), a_path, res_path)):
            c = pinv @ ph
            store_a.append(c)
            # residual variance over the aperture after removing j <= 1, 3, 10, 21
            store_r.append([np.mean((ph - Z[:, :J] @ np.linalg.lstsq(Z[:, :J], ph, rcond=None)[0]) ** 2) for J in (1, 3, 10, 21)])
    return np.array(a_single), np.array(res_single), np.array(a_path), np.array(res_path)


def validate_zernike(workers, N, dx, Ns, n_per=20):
    """Adds the Zernike-domain check to validation.json: the quantity that the Zernike channel model uses."""
    ctx = mp.get_context("forkserver")
    kids = np.random.SeedSequence(SEEDS["val_sf"] + 100).spawn(workers)
    out = {}
    for s2 in (1.0, 2.5):
        tasks = [(kids[i], N, dx, n_per, s2, Ns) for i in range(workers)]
        with ctx.Pool(workers) as pool:
            rs = pool.map(_zern_task, tasks)
        a1 = np.concatenate([r[0] for r in rs]); e1 = np.concatenate([r[1] for r in rs])
        a2 = np.concatenate([r[2] for r in rs]); e2 = np.concatenate([r[3] for r in rs])
        sc1 = (of.D_R / 1.0) ** (5 / 3)                    # single screens are generated with r0 = 1 m
        sc2 = of.turbulence(s2)["D_over_r0"] ** (5 / 3)     # path sum has the plane-wave r0 of the link
        noll = np.array([of.noll_cn(of.noll_nm(j)[0]) for j in range(2, 67)])
        nr = np.array([of.noll_nm(j)[0] for j in range(2, 67)])
        resid_th = [of.noll_residual(J) for J in (1, 3, 10, 21)]
        ent = {"n_screens": int(a1.shape[0]), "radial_orders": list(range(1, 11))}
        for name, a_, e_, sc in (("single_screen", a1, e1, sc1), ("path_sum", a2, e2, sc2)):
            var = np.mean(a_[:, 1:] ** 2, 0) / sc
            ent[name] = {"var_over_noll_by_radial_order": [float(np.sum(var[nr == k]) / np.sum(noll[nr == k])) for k in range(1, 11)],
                         "resid_J1_J3_J10_J21_over_noll": (np.mean(e_, 0) / sc / np.array(resid_th)).tolist()}
        out[str(s2)] = ent
        if s2 == 1.0:
            out["single_screen_note"] = "screens generated for r0 = 1 m; the ratio does not depend on r0"
        print("zernike check", s2, json.dumps(ent), flush=True)
    vpath = os.path.join(HERE, "validation.json")
    v = json.load(open(vpath)) if os.path.exists(vpath) else {}
    v["zernike_domain"] = out
    offs = np.array(v.get("structure_function", {}).get("r_m", [])) / dx
    if offs.size:
        Dg = sf_generator_expectation(N, dx, np.round(offs).astype(int))
        v["structure_function"]["generator_expectation_over_theory"] = (Dg / (6.88 * (offs * dx) ** (5 / 3))).tolist()
    with open(vpath, "w") as f:
        json.dump(v, f, indent=1)


def andrews_pw_si(s2):
    s65 = s2 ** 1.2
    return math.exp(0.49 * s2 / (1 + 1.11 * s65) ** (7 / 6) + 0.51 * s2 / (1 + 0.69 * s65) ** (5 / 6)) - 1


def gauss_beam_weak_si(s2):
    Th0, La0 = 1.0, 2 * L / (K * W0 ** 2)
    Th, La = Th0 / (Th0 ** 2 + La0 ** 2), La0 / (Th0 ** 2 + La0 ** 2)
    return 3.86 * s2 * (0.40 * ((1 + 2 * Th) ** 2 + 4 * La ** 2) ** (5 / 12) * math.cos(5 / 6 * math.atan((1 + 2 * Th) / (2 * La)))
                        - 11 / 16 * La ** (5 / 6))


def validate(workers, N, dx, Ns):
    t0 = time.time()
    ctx = mp.get_context("forkserver")
    v = {"sampling": sampling_checks((0.2, 1.0, 2.5), N, dx, Ns),
         "sampling_Ns20": sampling_checks((1.0, 2.5), N, dx, 20),
         "layout": {str(s): layout_summary(s, Ns) for s in (0.2, 1.0, 2.5)},
         "layout_Ns20": {str(s): layout_summary(s, 20) for s in (1.0, 2.5)}}
    # vacuum check: propagated LG modes against the analytic receive modes
    link = Link(1.0, N, dx, Ns)
    U = link.propagate(None, screens=False)
    rec = link.receive(U)
    ch = of.OAMChannel(modes=TX8, rx_modes=RX_ALL)
    vac_of = [float(np.sum(np.abs(of.lg_mode(m, ch.grid["r"], ch.grid["phi"], ch.w)) ** 2) * ch.grid["dA"]) for m in TX8]
    Gv = rec["G_raw"]
    ti = [RX_ALL.index(m) for m in TX8]
    v["vacuum"] = {"self_splitstep": [float(Gv[ti[i], i]) for i in range(8)], "self_oamfso_aperture_power": vac_of,
                   "self_oamfso_G": [x * x for x in vac_of],
                   "max_offdiag": float(np.max(Gv * (1 - np.eye(13)[:, ti]))), "Pgrid": rec["Pgrid"].tolist(),
                   "Pap": rec["Pap"].tolist(), "tilt_rad": rec["tilt"].tolist()}
    # (a) structure function of single screens (the weakest r0 used: sigma_R^2 = 2.5 screen r0 is irrelevant,
    # the screens are generated for r0 = 1 m and D scales exactly as r0^(-5/3); report D / 6.88 (r/r0)^(5/3))
    nsf = 400
    tasks = [(np.random.SeedSequence(SEEDS["val_sf"]).spawn(workers)[i], N, dx, nsf // workers + 1) for i in range(workers)]
    with ctx.Pool(workers) as pool:
        rs = pool.map(_sf_task, tasks)
    offs = rs[0][0]
    acc = sum(r[1] for r in rs); cnt = sum(r[2] for r in rs)
    Dsf = acc / cnt
    rr = offs * dx
    v["structure_function"] = {"n_screens": int(cnt), "r_m": rr.tolist(), "D_over_theory": (Dsf / (6.88 * rr ** (5 / 3))).tolist(),
                               "grid_side_m": N * dx, "note": "r0 = 1 m; D(r) scales as r0^(-5/3)"}
    print("structure function ratio:", np.round(Dsf / (6.88 * rr ** (5 / 3)), 3), flush=True)
    # (b) plane-wave scintillation over the central 0.2 m x 0.2 m on a 1024 x 1024 grid (dx as main run)
    v["plane_wave"] = {}
    for s2 in (0.2, 1.0, 2.5):
        npw = 480
        kids = np.random.SeedSequence([SEEDS["val_pw"], int(s2 * 10)]).spawn(workers)
        tasks = [(s2, 2 * N, dx, Ns, kids[i], npw // workers, False) for i in range(workers)]
        with ctx.Pool(workers) as pool:
            rs = pool.map(_pw_task, tasks)
        I = np.concatenate([r[0] for r in rs])
        Ivac = rs[0][1]
        si = float(np.mean(I ** 2) / np.mean(I) ** 2 - 1)
        per = np.mean(I ** 2, axis=(1, 2)) / np.mean(I) ** 2 - 1          # per-realization, for an error bar
        v["plane_wave"][str(s2)] = {"n_real": int(I.shape[0]), "grid": [2 * N, dx], "box_m": 0.2, "SI": si,
                                    "SI_se_between_realizations": float(per.std(ddof=1) / math.sqrt(per.size)),
                                    "theory_rytov": s2, "theory_andrews_phillips": andrews_pw_si(s2),
                                    "mean_I": float(I.mean()), "vacuum_I_range": [float(Ivac.min()), float(Ivac.max())]}
        print("plane wave", s2, v["plane_wave"][str(s2)], flush=True)
    # (c) Gaussian beam (beacon) on-axis scintillation, weak turbulence
    ngb = 2400
    kids = np.random.SeedSequence(SEEDS["val_gb"]).spawn(workers)
    tasks = [(0.2, N, dx, Ns, kids[i], ngb // workers, True) for i in range(workers)]
    with ctx.Pool(workers) as pool:
        rs = pool.map(_pw_task, tasks)
    x = np.concatenate([r[0] for r in rs])
    si = float(np.mean(x ** 2) / np.mean(x) ** 2 - 1)
    boot = np.random.default_rng(1).integers(0, x.size, (400, x.size))
    sib = np.mean(x[boot] ** 2, 1) / np.mean(x[boot], 1) ** 2 - 1
    v["gaussian_beam_on_axis"] = {"s2": 0.2, "n_real": int(x.size), "SI": si, "SI_boot_se": float(sib.std()),
                                  "theory_weak_rytov": gauss_beam_weak_si(0.2), "mean_I_over_vac": float(x.mean())}
    print("gaussian beam", v["gaussian_beam_on_axis"], flush=True)
    v["runtime_s"] = time.time() - t0
    with open(os.path.join(HERE, "validation.json"), "w") as f:
        json.dump(v, f, indent=1, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else (bool(o) if isinstance(o, np.bool_) else float(o)))
    return v


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("validate", "validate_zernike", "run", "zern", "analyze", "time"))
    ap.add_argument("--s2", type=float, default=1.0)
    ap.add_argument("--N", type=int, default=20000, help="number of realizations")
    ap.add_argument("--nx", type=int, default=N_DEF, help="grid points per side")
    ap.add_argument("--dx", type=float, default=DX_DEF)
    ap.add_argument("--Ns", type=int, default=NS_DEF)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--tag", default="ss")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--nsub", type=int, default=3, help="subharmonic levels of the phase screens")
    a = ap.parse_args()
    if a.cmd == "validate":
        validate(a.workers, a.nx, a.dx, a.Ns)
    elif a.cmd == "validate_zernike":
        validate_zernike(a.workers, a.nx, a.dx, a.Ns)
    elif a.cmd == "time":
        out = {}
        for dt in (np.complex64, np.complex128):
            link = Link(a.s2, a.nx, a.dx, a.Ns, dt)
            rng = np.random.default_rng(0)
            t = time.time()
            out[dt] = [link.receive(link.propagate(rng)) for _ in range(3)]
            print(f"{np.dtype(dt).name}: {(time.time() - t) / 3:.3f} s per realization (single process)")
        for v in ("G_raw", "G_tt", "G_ttc"):
            d = max(float(np.max(np.abs(x[v] - y[v]))) for x, y in zip(out[np.complex64], out[np.complex128]))
            mn = min(float(np.min(y[v])) for y in out[np.complex128])
            print(f"{v}: max |G(complex64) - G(complex128)| = {d:.2e}  (smallest entry {mn:.2e})")
    elif a.cmd == "run":
        seed = a.seed if a.seed is not None else SEEDS[a.tag if a.tag in ("ss20", "gridfine", "gridwide", "ss6sh") else "ss"][a.s2]
        run_splitstep(a.s2, a.N, a.nx, a.dx, a.Ns, seed, a.workers, a.tag, a.nsub)
    elif a.cmd == "zern":
        seed = a.seed if a.seed is not None else SEEDS["zernbig" if a.tag == "zernbig" else "zern"][a.s2]
        run_zernike(a.s2, a.N, seed, a.tag)
    elif a.cmd == "analyze":
        analyze(a.s2)
