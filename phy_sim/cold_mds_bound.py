"""Erasure code across the streams at a cold start and for the anchor payload, from the stored failure traces.

Cold start (Model B, six-slot windows as in codec/eval_cold.py): an (M, M/2) MDS code across the M streams at twice the
payload sends the enhancement layer of the full view in slots 1 to 3, like pairwise replication; it recovers a slot
with at most M/2 failed streams. The probability that it fails in at least one of slots 1 to 3 is an upper bound on the
probability that the view fails with this code (with or without concealment), since every such window is counted as
failed; the windows in which all anchor copies fail in slot 1 or 2 are added, since the view then fails as well. Anchor payload (Model A, modeset_2.5.json): an (M, M/n_A) code at n_A times the payload fails if more than
M - M/n_A streams fail, against the joint loss of the n_A replicated modes.
Output: results/cold_mds_bound.json"""
import glob
import itertools
import json

import numpy as np

out = {"cold_start": {}, "anchor": {}}
for tau in ("2.5", "10"):
    for cfg, M in (("T8R8", 8), ("T6R11", 6)):
        xs = [np.load(f)[cfg] for f in sorted(glob.glob(f"results/correlated/patterns/traces_corr_2.5_{tau}ms_pat*.npz"))]
        bits = np.unpackbits(np.concatenate(xs)[..., None], axis=-1, bitorder="little")[..., :M].astype(bool)
        K = bits.sum(-1); nW = K.shape[1] - 6
        nA = 4 if cfg == "T8R8" else 2                        # anchor copies as in eval_cold.py
        flat = bits.reshape(-1, M)
        anchor = min(itertools.combinations(range(M), nA), key=lambda s_: int(flat[:, list(s_)].all(1).sum()))
        alost = bits[..., list(anchor)].all(-1)                # all anchor copies lost in a slot
        fail_code = np.zeros((K.shape[0], nW), bool)
        for t in range(3):
            fail_code |= K[:, t:nW + t] > M // 2
        fail_anchor = alost[:, :nW] | alost[:, 1:nW + 1]       # anchor payload in slots 1 and 2
        fail = fail_code | fail_anchor
        out["cold_start"][f"{cfg}_tau{tau}"] = {"code": [M, M // 2], "p_fail_bound": float(fail.mean()), "p_code_fails": float(fail_code.mean()),
                                                 "p_anchor_lost": float(fail_anchor.mean()), "windows": int(fail.size), "anchor_modes": [int(x) for x in anchor]}
ms = json.load(open("results/modeset_2.5.json"))["configs"]
for cfg in ("T8R8", "T6R11", "T6R8"):
    cf = ms[cfg]; h = np.array(cf["nfail_grid"][cf["snr_labels"].index("op")], float); N = h.sum(); M = len(cf["tx"])
    out["anchor"][cfg] = {f"({M},{M // n})": float(h[M - M // n + 1:].sum() / N) for n in (2, 4) if M % n == 0}
    out["anchor"][cfg]["p_out"] = float(1 - h[0] / N)
json.dump(out, open("results/cold_mds_bound.json", "w"), indent=1)
print(json.dumps(out, indent=1))
