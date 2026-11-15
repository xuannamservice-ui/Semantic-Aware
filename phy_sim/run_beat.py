"""Common-carrier variant of the OAM-FSO link (sensitivity analysis).

The main model gives every stream its own optical carrier, spaced by more than 2B, so that the beat terms between
streams fall outside the receiver filter and IM/DD detection is linear. Here all streams use lasers at one nominal
wavelength with independent phases, as needed for spectral reuse by OAM multiplexing. Port r then also detects the
signal-signal beat terms 2 Re(g_rn conj(g_rm) e^{j theta_nm}) sqrt(P_n P_m x_n x_m) of every mode pair (n, m), with
g_rn = h_rn sqrt(I_n) the complex coupling including scintillation and theta_nm a uniform random phase. They are
treated as zero-mean Gaussian interference with covariance, in units of the thermal-plus-shot noise variance,
    K_b[r, r'] = 2 snr0 sum_{n<m} Re( g_rn conj(g_rm) conj(g_r'n) g_r'm ),
which the MMSE equalizer sees as coloured noise (whitened by the Cholesky factor of I + K_b).
Common random numbers with the main model; output results/beat_<s2>.json.
"""
import json, math, os, sys, time
import numpy as np
import oamfso as of
import run_modeset as rm

HERE = os.path.dirname(os.path.abspath(__file__))
CFGS = ("T8R8", "T8R11", "T8R13", "T6R8", "T6R11", "T4R8")


def main(s2, N=200_000, batch=500, seed=606):
    t0 = time.time()
    base = of.OAMChannel()
    gr = base.grid
    U = {m: of.lg_mode(m, gr["r"], gr["phi"], base.w) for m in rm.RX_ALL}
    Ur = np.stack([U[m] for m in rm.RX_ALL]); Ut = np.stack([U[m] for m in rm.TX8])
    Q = (np.conj(Ur)[:, None, :] * Ut[None, :, :] * gr["dA"]).reshape(len(rm.RX_ALL) * len(rm.TX8), -1).T
    Qr, Qi = np.ascontiguousarray(Q.real), np.ascontiguousarray(Q.imag)
    rng = np.random.default_rng(seed)
    VAR = ("separate", "W4", "W2", "common")             # carriers: one per mode, 4, 2, one (carrier of mode l: l mod W)
    cnt = {k: {v: 0 for v in VAR} for k in CFGS}
    nf = {k: {v: np.zeros(9, np.int64) for v in VAR} for k in CFGS}
    done = 0
    while done < N:
        b = min(batch, N - done)
        z = rng.standard_normal((b, base.j_list.size))
        Iall = base.scint(rng, b, s2)
        Phi = base.coeffs(z, s2) @ base.Z
        c, s = np.cos(Phi), np.sin(Phi)
        h = ((c @ Qr - s @ Qi) + 1j * (c @ Qi + s @ Qr)).reshape(b, len(rm.RX_ALL), len(rm.TX8))
        gall = h * np.sqrt(Iall)[:, None, :]                       # complex coupling with scintillation
        for k in CFGS:
            tx, rx = rm.CFG[k]
            g = gall[:, [rm.RX_ALL.index(m) for m in rx]][:, :, [rm.TX8.index(m) for m in tx]]
            M, Nr = len(tx), len(rx)
            snr = rm.snr0(M)
            G = np.abs(g) ** 2
            fails = {"separate": of.sinr_mmse(G, snr) < rm.GTH}
            for v, W in (("W4", 4), ("W2", 2), ("common", 1)):
                car = [int(l) % W for l in tx]
                Kb = np.zeros((b, Nr, Nr))
                for n in range(M):
                    for m in range(n + 1, M):
                        if car[n] != car[m]:
                            continue                 # different carriers: beat outside the receiver band
                        a = g[:, :, n] * np.conj(g[:, :, m])
                        Kb += np.real(a[:, :, None] * np.conj(a[:, None, :]))
                L = np.linalg.cholesky(np.eye(Nr)[None] + 2.0 * snr * Kb)
                fails[v] = of.sinr_mmse(np.linalg.solve(L, G), snr) < rm.GTH
            for v in VAR:
                cnt[k][v] += int(fails[v].any(1).sum())
                nf[k][v] += np.bincount(fails[v].sum(1), minlength=9)[:9]
        done += b
        if done % 20000 == 0:
            print(f"s2={s2} {done}/{N} {time.time()-t0:.0f}s", flush=True)
    res = {"s2": s2, "N": N, "seed": seed, "carrier_of_mode": "l mod W", "configs": {}}
    for k in CFGS:
        res["configs"][k] = {v: {"outage": cnt[k][v] / N, "count": cnt[k][v], "ci95": rm_ci(cnt[k][v], N),
                                 "nfail": nf[k][v].tolist()} for v in VAR}
        print(k, " ".join(f"{v}={cnt[k][v] / N:.3g}" for v in VAR), flush=True)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(res, open(os.path.join(HERE, "results", f"beat_{s2}.json"), "w"), indent=1)
    print(f"done in {time.time()-t0:.0f}s", flush=True)


def rm_ci(k, n):
    p = k / n; h = 1.96 * math.sqrt(max(p * (1 - p), 1e-300) / n)
    return [max(0.0, p - h), p + h]


if __name__ == "__main__":
    main(float(sys.argv[1]), N=int(sys.argv[2]) if len(sys.argv) > 2 else 200_000)
