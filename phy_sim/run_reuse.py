"""Wavelength reuse in weak turbulence: M OAM modes on W < M optical carriers (carrier of mode l: l mod W), IM/DD.

Modes on the same carrier beat inside the receiver band; the beat terms are Gaussian interference with the covariance
of run_beat.py, seen by the MMSE equalizer as coloured noise. Records, per configuration, the outage, the per-mode and
per-pattern failure counts, the number of failed streams and the failure patterns of the first 1e6 slots (trace for the
codec evaluation, bit i = tx[i] failed).
usage: python run_reuse.py chunk <s2> <seed> <N> <out.npz>   |   python run_reuse.py merge <s2> <tag> <chunk.npz>...
"""
import itertools, json, os, sys, time
import numpy as np
import oamfso as of
import run_modeset as rm

CFGW = {"T8R11W4": ("T8R11", 4), "T6R11W4": ("T6R11", 4)}


def chunk(s2, seed, N, out, batch=500, n_trace=1_000_000):
    t0 = time.time(); base = of.OAMChannel(); gr = base.grid
    U = {m: of.lg_mode(m, gr["r"], gr["phi"], base.w) for m in rm.RX_ALL}
    Ur = np.stack([U[m] for m in rm.RX_ALL]); Ut = np.stack([U[m] for m in rm.TX8])
    Q = (np.conj(Ur)[:, None, :] * Ut[None, :, :] * gr["dA"]).reshape(len(rm.RX_ALL) * len(rm.TX8), -1).T
    Qr, Qi = np.ascontiguousarray(Q.real), np.ascontiguousarray(Q.imag)
    rng = np.random.default_rng(seed); res = {}
    for k, (c, W) in CFGW.items():
        M = len(rm.CFG[c][0])
        res[k] = {"pat": np.zeros(2 ** M, np.int64), "trace": np.zeros(min(N, n_trace), np.uint8)}
    done = 0
    while done < N:
        b = min(batch, N - done)
        z = rng.standard_normal((b, base.j_list.size)); Iall = base.scint(rng, b, s2)
        Phi = base.coeffs(z, s2) @ base.Z; cs, sn = np.cos(Phi), np.sin(Phi)
        h = ((cs @ Qr - sn @ Qi) + 1j * (cs @ Qi + sn @ Qr)).reshape(b, len(rm.RX_ALL), len(rm.TX8))
        gall = h * np.sqrt(Iall)[:, None, :]
        for k, (c, W) in CFGW.items():
            tx, rx = rm.CFG[c]; M, Nr = len(tx), len(rx); snr = rm.snr0(M)
            g = gall[:, [rm.RX_ALL.index(m) for m in rx]][:, :, [rm.TX8.index(m) for m in tx]]
            G = np.abs(g) ** 2; car = [int(l) % W for l in tx]
            Kb = np.zeros((b, Nr, Nr))
            for n in range(M):
                for m in range(n + 1, M):
                    if car[n] == car[m]:
                        a = g[:, :, n] * np.conj(g[:, :, m]); Kb += np.real(a[:, :, None] * np.conj(a[:, None, :]))
            L = np.linalg.cholesky(np.eye(Nr)[None] + 2.0 * snr * Kb)
            f = of.sinr_mmse(np.linalg.solve(L, G), snr) < rm.GTH
            code = (f.astype(np.int64) << np.arange(M)).sum(1)
            res[k]["pat"] += np.bincount(code, minlength=2 ** M)
            if done < res[k]["trace"].size:
                e = min(done + b, res[k]["trace"].size); res[k]["trace"][done:e] = code[:e - done]
        done += b
        if done % 100000 == 0:
            print(f"seed {seed}: {done}/{N} {time.time()-t0:.0f}s", flush=True)
    np.savez(out, N=N, seed=seed, **{f"{k}_pat": v["pat"] for k, v in res.items()}, **{f"{k}_trace": v["trace"] for k, v in res.items()})


def merge(s2, tag, files):
    parts = [np.load(f) for f in files]; N = int(sum(int(p["N"]) for p in parts))
    out = {"s2": float(s2), "N": N, "seeds": [int(p["seed"]) for p in parts], "carrier_of_mode": "l mod W", "configs": {}}
    traces = {}
    for k, (c, W) in CFGW.items():
        tx, rx = rm.CFG[c]; M = len(tx)
        pat = sum(p[f"{k}_pat"] for p in parts)
        codes = np.arange(2 ** M); bits = (codes[:, None] >> np.arange(M)) & 1
        per_mode = (bits * pat[:, None]).sum(0); nf = np.bincount(bits.sum(1), weights=pat, minlength=M + 1)
        best = {}
        for n in range(1, 5):
            cand = []
            for S in itertools.combinations(range(M), n):
                mask = sum(1 << i for i in S); cand.append((int(pat[(codes & mask) == mask].sum()), [int(tx[i]) for i in S]))
            cnt, modes = min(cand); best[n] = {"modes": modes, "count": cnt, "prob": cnt / N}
        out["configs"][k] = {"base_config": c, "W": W, "tx": list(tx), "rx": list(rx), "carriers": [int(l) % W for l in tx],
                             "snr_labels": ["op"], "per_mode_count": [per_mode.astype(int).tolist()],
                             "outage": float(pat[1:].sum() / N), "outage_count": int(pat[1:].sum()),
                             "nfail": nf.astype(int).tolist(), "patterns_nonzero": int((pat[1:] > 0).sum()),
                             "mean_K": float((np.arange(M + 1) * nf).sum() / max(1, nf[1:].sum())), "anchor_best": best,
                             "pattern_hist": pat.astype(int).tolist()}
        traces[k] = parts[0][f"{k}_trace"]
        print(k, f"outage {out['configs'][k]['outage']:.3g} ({out['configs'][k]['outage_count']}), E[K|out] {out['configs'][k]['mean_K']:.2f}, "
              f"pi_A(2) {best[2]['prob']:.2g} {best[2]['modes']}, pi_A(4) {best[4]['prob']:.2g}")
    os.makedirs("results", exist_ok=True)
    json.dump(out, open(f"results/modeset_{tag}.json", "w"), indent=1)
    np.savez_compressed(f"results/traces_{tag}.npz", **traces)


if __name__ == "__main__":
    if sys.argv[1] == "chunk":
        chunk(float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5])
    else:
        merge(sys.argv[2], sys.argv[3], sys.argv[4:])
