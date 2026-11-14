"""Per-realization failure patterns of the split-step channel (input of the codec evaluation).

For every stored split-step realization (raw_ss_<s2>.npz, coupling matrices G of the 8 transmitted modes on the 13
sorter ports) and every configuration, the MMSE SINR of each transmitted stream at the link-budget operating point is
compared with the HD-FEC threshold; bit i of the pattern is set if the i-th transmitted mode of the configuration failed
(same convention as run_modeset.py traces). Output: patterns_<variant>_<s2>.npz with one uint8 array per configuration
and the per-mode failure counts, in the format read by the codec (traces_<s2>.npz, modeset_<s2>.json).
usage: python ss_patterns.py --s2 2.5 --variant tt
"""
import argparse
import json

import numpy as np

import split_step as ss

of, rm = ss.of, ss.rm


def main(a):
    z = np.load(f"raw_ss_{a.s2:g}.npz")
    G = z[f"G_{a.variant}"]
    out, meta = {}, {}
    for k in ("T8R8", "T6R8", "T6R11", "T8R11"):
        tx, rx = rm.CFG[k]
        ridx = [ss.RX_ALL.index(m) for m in rx]; tidx = [ss.TX8.index(m) for m in tx]
        if any(m not in ss.TX8 for m in tx):
            continue
        lam, V2 = of.gram_eig(G[:, ridx][:, :, tidx].astype(np.float64))
        fail = of.sinr_mmse_eig(lam, V2, rm.snr0(len(tx))) < ss.GTH
        out[k] = (fail * (1 << np.arange(len(tx)))).sum(1).astype(np.uint8)
        meta[k] = {"tx": [int(x) for x in tx], "rx": [int(x) for x in rx], "n": int(len(fail)),
                   "outage": float(fail.any(1).mean()), "per_mode_count": fail.sum(0).tolist()}
        print(k, meta[k]["outage"], meta[k]["per_mode_count"], flush=True)
    np.savez_compressed(f"patterns_{a.variant}_{a.s2:g}.npz", **out)
    json.dump(meta, open(f"patterns_{a.variant}_{a.s2:g}.json", "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--s2", type=float, default=2.5)
    ap.add_argument("--variant", default="tt", choices=("raw", "tt", "ttc"))
    main(ap.parse_args())
