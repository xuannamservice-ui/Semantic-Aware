"""Repetitions of the MLP rows of Table tab:latency (bench_latency.py --only-mlp 1, same harness: 250,000 anchors, random
weights, FP16, batches of 16,384, 50 warm-up + 2,000 timed runs per row, idle GPU, one process per repetition) merged
into results/bench_latency_rev6.json: stages <row>_rep = median over the repetitions of the per-repetition means (with the
means, medians and 99th percentiles of every repetition); mlp_rows_median = the same values per row and L, plus the
launch-bound check of the profiling run (host enqueue time and GPU kernel time per call). The earlier single-run keys
are kept.
usage: python merge_mlp_reps.py results/bench_latency_rev6_mlprep1.json ... --profile results/bench_latency_rev6_mlpprof.json
"""
import argparse
import json

import numpy as np


def main(a):
    reps = [json.load(open(f)) for f in a.reps]
    d = json.load(open(a.target))
    rows = sorted({k for r in reps for k in r["stages"]})
    med = {}
    for k in rows:
        xs = [r["stages"][k] for r in reps if k in r["stages"]]
        means = [x["mean"] for x in xs]
        v = {"mean": float(np.median(means)), "means": means, "p50s": [x["p50"] for x in xs], "p99s": [x["p99"] for x in xs],
             "min_mean": float(min(means)), "max_mean": float(max(means)), "n_reps": len(xs), "n_timed_each": xs[0]["n"]}
        med[k] = v
        if k.startswith(("mlp_fp16_L", "mlp_fp16_compiled_L")):
            d["stages"][f"{k}_rep"] = v
    block = {"semantics": __doc__.split("usage")[0], "files": a.reps, "rows": med}
    if a.profile:
        p = json.load(open(a.profile))
        block["profile_file"] = a.profile
        block["launch_bound_check"] = p.get("mlp_profile", {})
        block["profile_run_means"] = {k: v["mean"] for k, v in p["stages"].items()}
    d["mlp_rows_median"] = block
    with open(a.target, "w") as fh:
        json.dump(d, fh, indent=1)
    for k, v in med.items():
        print(f"{k:28s} median of means {v['mean']:.3f}  means {[round(x, 3) for x in v['means']]}  p99s {[round(x, 3) for x in v['p99s']]}")
    for k, v in block.get("launch_bound_check", {}).items():
        print(k, {kk: vv for kk, vv in v.items() if kk != "top_kernels"})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("reps", nargs="+")
    ap.add_argument("--profile", default="")
    ap.add_argument("--target", default="results/bench_latency_rev6.json")
    main(ap.parse_args())
