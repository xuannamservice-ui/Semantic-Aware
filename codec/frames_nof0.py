"""Frame-level results (eval_frames.py, Table tab:frames) without frame 0 of every frame path.

Frame 0 of a path is the full-view start-up frame (the edge cache is empty, the whole view is sent in one slot); it is a
cold start and is evaluated as such (eval_cold4.py). This script recomputes the summary block of every
results/<scene>/frames_v2a_r1_<cfg>_tau<t>_eps0.166667[_a35r].json from its per-frame rows without the rows of path
frame 0 (row field "frame" = position on the capture path, 0 = first frame of the chain) and writes
results/<scene>/frames_v2a_r1_<cfg>_tau<t>_eps0.166667[_a35r]_nof0.json (same structure: metadata, summary, rows; plus
"excluded": "frame 0"). The summary has the semantics of eval_frames.py (psi capped at 99 dB, p_fail = share of frames
below the threshold, all chains of the file).
It also writes results/rev6_frames_nof0.json with pooled counts over the scenes, with and without frame 0, and for
frame 0 alone:
  files : per (cfg, tau, variant) over all chains of every scene file (the semantics of the summary blocks);
  table : the semantics of Table tab:frames (fill.py): T8R8, chains 0-9, none / conceal / MDS from the main runs and
          ARQ / ARQ+conceal from the *_a35r runs (retransmission in slots 3 and 5, remapped), frames where all six
          schemes exist.
usage: python frames_nof0.py [--out results]
"""
import argparse
import glob
import json
import os
import re

import numpy as np

SCH = ["none", "conceal", "arq", "arq+conceal", "mds", "mds+conceal"]


def summary(rows, schemes, ths):
    out = {}
    for sch in schemes:
        rs = [x for x in rows if x["scheme"] == sch]
        if not rs:
            continue
        psi = np.array([min(x["psi"], 99.0) for x in rs])
        out[sch] = {"frames": len(rs), "frames_with_missing": int(sum(x["n_missing"] > 0 for x in rs)),
                    "psnr": float(np.mean([x["psnr"] for x in rs])), "lpips": float(np.mean([x["lpips"] for x in rs])),
                    **{f"p_fail_psi{t}": float((psi < t).mean()) for t in ths},
                    "psi_min": float(psi.min()), "psi_p01": float(np.percentile(psi, 1))}
    return out


def counts(rows, sch):
    rs = [x for x in rows if x["scheme"] == sch]
    return {"frames": len(rs), "paths": len({(x.get("scene"), x["chain"]) for x in rs}),
            "with_missing": int(sum(x["n_missing"] > 0 for x in rs)),
            "below30": int(sum(x["psi"] < 30 for x in rs)), "below35": int(sum(x["psi"] < 35 for x in rs)),
            "psi_min": float(min((min(x["psi"], 99.0) for x in rs), default=float("nan")))}


def main(a):
    files = sorted(f for f in glob.glob(os.path.join(a.out, "*", "frames_v2a_r1_*_eps0.166667*.json"))
                   if re.search(r"frames_v2a_r1_[^_]+_tau[0-9.]+_eps0\.166667(_a35r)?\.json$", f))
    pooled, table_rows = {}, {}
    for f in files:
        d = json.load(open(f))
        ths = sorted(float(k.split("psi")[1]) for k in next(iter(d["summary"].values())) if k.startswith("p_fail_psi"))
        schemes = list(d["summary"])
        rows = d["rows"]
        nof0 = [r for r in rows if r["frame"] != 0]
        out = {k: v for k, v in d.items() if k not in ("summary", "rows")}
        out.update(excluded="frame 0 of every path (full-view start-up frame)", summary=summary(nof0, schemes, ths),
                   summary_with_frame0=d["summary"], frame0_rows=sum(r["frame"] == 0 for r in rows), rows=nof0)
        with open(f[:-5] + "_nof0.json", "w") as fh:
            json.dump(out, fh)
        m = re.search(r"frames_v2a_r1_([^_]+)_tau([0-9.]+)_eps0\.166667(_a35r)?\.json$", f)
        key = f"{m.group(1)}|tau{m.group(2)}|{'a35r' if m.group(3) else 'main'}"
        sc = d["scene"]
        g = pooled.setdefault(key, {"scenes": [], "rows": []})
        g["scenes"].append(sc); g["rows"] += [dict(r, scene=sc) for r in rows]
        if m.group(1) == "T8R8":
            table_rows.setdefault((m.group(2), sc), {})["a35r" if m.group(3) else "main"] = rows
        print(f, len(rows), "rows,", out["frame0_rows"], "in frame 0", flush=True)
    res = {"semantics": __doc__.split("usage")[0], "files": {}, "table": {}}
    for key, g in sorted(pooled.items()):
        R = g["rows"]; schemes = [s for s in SCH if any(r["scheme"] == s for r in R)]
        res["files"][key] = {"scenes": sorted(g["scenes"]),
                             "with_frame0": {s: counts(R, s) for s in schemes},
                             "without_frame0": {s: counts([r for r in R if r["frame"] != 0], s) for s in schemes},
                             "frame0_only": {s: counts([r for r in R if r["frame"] == 0], s) for s in schemes}}
    for tau in sorted({k[0] for k in table_rows}):
        D = {}
        sc_ok = [sc for (t, sc), v in table_rows.items() if t == tau and "main" in v]
        a35 = all("a35r" in table_rows[(tau, sc)] for sc in sc_ok)
        for sc in sc_ok:
            v = table_rows[(tau, sc)]
            for r in v["main"]:
                if r["chain"] < 10 and not (a35 and r["scheme"] in ("arq", "arq+conceal")):
                    D.setdefault((sc, r["chain"], r["view"], r["frame"]), {})[r["scheme"]] = dict(r, scene=sc)
            if a35:
                for r in v["a35r"]:
                    if r["chain"] < 10:
                        D.setdefault((sc, r["chain"], r["view"], r["frame"]), {})[r["scheme"]] = dict(r, scene=sc)
        keys = [k for k in D if len(D[k]) == 6]
        R = [D[k][s] for k in keys for s in SCH]
        res["table"][f"T8R8|tau{tau}"] = {"scenes": sorted(sc_ok), "arq_from_a35r": a35, "chains": "0-9",
                                         "with_frame0": {s: counts(R, s) for s in SCH},
                                         "without_frame0": {s: counts([r for r in R if r["frame"] != 0], s) for s in SCH},
                                         "frame0_only": {s: counts([r for r in R if r["frame"] == 0], s) for s in SCH}}
    with open(os.path.join(a.out, "rev6_frames_nof0.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    for k, v in res["table"].items():
        print(k)
        for part in ("with_frame0", "without_frame0", "frame0_only"):
            print("  ", part, {s: (c["frames"], c["paths"], c["with_missing"], c["below30"], c["below35"]) for s, c in v[part].items()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    main(ap.parse_args())
