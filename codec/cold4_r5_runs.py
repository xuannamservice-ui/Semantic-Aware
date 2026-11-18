"""One-frame (_r5) column of the executed cold start (eval_cold4.py) over repeated runs: the run of rev6_cold4_summary.json
(suffix "", margin 0 ms, chunk 8192) and the repeats with suffixes _run2 and _run3 (same windows, same margin and chunk,
executed again with a new calibration), plus the run with a positive safety margin (suffix _m05, margin 0.5 ms).
Every run is pooled exactly as eval_cold4.py --aggregate (undeliverable views left out per run, mean over the views of a
scene, then over the scenes). Output results/rev6_cold4_r5_runs.json:
  runs:    {run: {configs: {<cfg>|tau<t>|_r5: ...}, table_cold_30dB, table_cold_35dB, table_columns}} (structure of
           rev6_cold4_summary.json, one-frame columns only)
  median / min / max: table_cold_30dB and table_cold_35dB per entry over the margin-0 runs (run1, run2, run3)
  margin_05: the margin-0.5 run in the same structure
  always_fail_views: per run and column, the deliverable views whose loss-free schedule of replication/MDS (S_R slots)
           or of the plain schedule (S_A slots) exceeds the deadline, with the executed loss-free totals
  pooled_feasibility: the same tables with the feasibility of every view and schedule decided once from all loss-free
           executions of that schedule (median over the 4 runs x 3 executions x both tau_c, and for the S_A schedule,
           which does not depend on the configuration, over both configurations as well; the margin does not enter a
           loss-free execution, so the margin-0.5 run contributes), instead of the median of 3 executions of one run; a
           view that one run classified infeasible but the pooled median finds feasible takes the executed windows of
           the same view from the other runs (same windows); per run, then median / min / max over run1-run3
usage: python cold4_r5_runs.py
"""
import argparse
import json
import os

import numpy as np

import eval_cold4 as ec4

ALLS = ("garden", "bicycle", "stump", "room", "counter", "kitchen", "bonsai", "truck", "train", "playroom", "drjohnson")
CFGS = (("T8R8", "2.5"), ("T8R8", "10"), ("T6R11", "2.5"), ("T6R11", "10"))
RUNS = {"run1": "", "run2": "_run2", "run3": "_run3"}


def r5_only(out):
    cols = [c for c in out["table_columns"] if c.endswith("_r5")]
    res = {"configs": {k: v for k, v in out["configs"].items() if k.endswith("_r5")}, "table_columns": cols}
    for th in (30, 35):
        res[f"table_cold_{th}dB"] = {sch: {c: r[c] for c in cols} for sch, r in out[f"table_cold_{th}dB"].items()}
    return res


def always_fail_views(a, sfx):
    out = {}
    for cfg, tau in CFGS:
        lst = []
        for s in ALLS:
            f = os.path.join(a.out, s, f"cold4{a.tag}_{cfg}_tau{tau}_r5{sfx}.json")
            if not os.path.exists(f):
                continue
            j = json.load(open(f))
            n_deliv = sum(not v["always_fail"]["none"] for v in j["views"])
            for v in j["views"]:
                af = v["always_fail"]
                if not af["none"] and any(af.values()):
                    lst.append({"scene": s, "view": v["view"], "S_A": v["S_A"], "S_R": v["S_R"], "n_visible": v["n_visible"],
                                "schemes": [k for k, x in af.items() if x], "free_A_totals": v["free_A"]["totals"], "free_R_totals": v["free_R"]["totals"],
                                "weight_in_column": 1.0 / (n_deliv * len(ALLS))})
        out[f"{cfg}|tau{tau}|_r5"] = lst
    return out


def pooled_tables(a):
    """Tables with the feasibility of each view decided from the pooled loss-free executions (see the module doc)."""
    sfxs = {**RUNS, "margin_05": "_m05"}
    J = {(r, cfg, tau, s): json.load(open(os.path.join(a.out, s, f"cold4{a.tag}_{cfg}_tau{tau}_r5{sx}.json")))
         for r, sx in sfxs.items() for cfg, tau in CFGS for s in ALLS
         if os.path.exists(os.path.join(a.out, s, f"cold4{a.tag}_{cfg}_tau{tau}_r5{sx}.json"))}
    poolA, poolR = {}, {}
    for (r, cfg, tau, s), j in J.items():
        for v in j["views"]:
            poolA.setdefault((s, v["view"]), []).extend(v["free_A"]["totals"])
            poolR.setdefault((s, cfg, v["view"]), []).extend(v["free_R"]["totals"])
    TF = 11.1
    infA = {k: float(np.median(x)) > TF for k, x in poolA.items()}
    infR = {k: float(np.median(x)) > TF for k, x in poolR.items()}
    out = {"n_executions_A": {f"{k[0]}:{k[1]}": len(x) for k, x in poolA.items() if 10.5 < np.median(x) < 11.7},
           "borderline_views": {}, "runs": {}, "substituted_rows": {}}
    for k, x in poolA.items():
        if 10.8 < np.median(x) < 11.4:
            out["borderline_views"][f"{k[0]}:{k[1]}:A"] = {"median": float(np.median(x)), "min": float(np.min(x)), "max": float(np.max(x)), "n": len(x),
                                                            "frac_above": float(np.mean(np.asarray(x) > TF))}
    for k, x in poolR.items():
        if 10.8 < np.median(x) < 11.4:
            out["borderline_views"][f"{k[0]}:{k[1]}:{k[2]}:R"] = {"median": float(np.median(x)), "min": float(np.min(x)), "max": float(np.max(x)),
                                                                   "n": len(x), "frac_above": float(np.mean(np.asarray(x) > TF))}
    for r in sfxs:
        tabs = {th: {} for th in (30, 35)}
        nsub = 0
        for cfg, tau in CFGS:
            col = f"{cfg}|tau{tau}|_r5"; pr = {}
            for s in ALLS:
                j = J.get((r, cfg, tau, s))
                if j is None:
                    continue
                views, rows = [], [x for x in j["rows"]]
                for v in j["views"]:
                    v2 = dict(v); fa, fr = infA[(s, v["view"])], infR[(s, cfg, v["view"])]
                    v2["always_fail"] = {k: (fr if k.startswith(("rep2", "mds")) else fa) for k in v["always_fail"]}
                    views.append(v2)
                    for sch, was in v["always_fail"].items():          # infeasible in this run, feasible pooled: windows of another run
                        if was and not v2["always_fail"][sch]:
                            rows = [x for x in rows if not (x["view"] == v["view"] and x["scheme"] == sch)]
                            for r2 in sfxs:
                                j2 = J.get((r2, cfg, tau, s))
                                if j2 is not None and r2 != r and not next(w for w in j2["views"] if w["view"] == v["view"])["always_fail"][sch]:
                                    rows += [x for x in j2["rows"] if x["view"] == v["view"] and x["scheme"] == sch]; nsub += 1
                                    break
                xv = {v["view"] for v in views if v["always_fail"]["none"]}
                sm = ec4.summarize([x for x in rows if x["view"] not in xv], [v for v in views if v["view"] not in xv])
                for k, v in sm.items():
                    pr.setdefault(k, []).append(v)
                pr.setdefault("_undeliv", []).append((len(xv), len(views)))
            for th in (30, 35):
                for sch in ec4.SCHEMES[:6] + ("adaptive",) + ec4.SCHEMES[6:]:
                    tabs[th].setdefault(sch, {})[col] = float(np.mean(pr[f"{sch}/{th}"])) if f"{sch}/{th}" in pr else None
                u = pr.get("_undeliv", [])
                tabs[th].setdefault("undeliverable_views_frac", {})[col] = sum(x[0] for x in u) / max(1, sum(x[1] for x in u))
        out["runs"][r] = {f"table_cold_{th}dB": tabs[th] for th in (30, 35)}
        out["substituted_rows"][r] = nsub
    cols = [f"{c}|tau{t}|_r5" for c, t in CFGS]
    for th in (30, 35):
        k = f"table_cold_{th}dB"
        for stat, fn in (("median", np.median), ("min", np.min), ("max", np.max)):
            out.setdefault(stat, {})[k] = {sch: {c: float(fn([out["runs"][r][k][sch][c] for r in RUNS])) for c in cols} for sch in out["runs"]["run1"][k]}
    out["table_columns"] = cols
    return out


def main(a):
    res = {"semantics": __doc__.split("usage")[0], "runs": {}, "always_fail_views": {}}
    for name, sfx in list(RUNS.items()) + [("margin_05", "_m05")]:
        ns = argparse.Namespace(out=a.out, tag=a.tag, suffix=sfx)
        res["runs"][name] = r5_only(ec4.aggregate(ns, write=False))
        res["always_fail_views"][name] = always_fail_views(a, sfx)
    cols = res["runs"]["run1"]["table_columns"]
    for th in (30, 35):
        k = f"table_cold_{th}dB"
        for stat, fn in (("median", np.median), ("min", np.min), ("max", np.max)):
            tab = {}
            for sch in res["runs"]["run1"][k]:
                tab[sch] = {}
                for c in cols:
                    xs = [res["runs"][r][k][sch][c] for r in RUNS if res["runs"][r][k][sch].get(c) is not None]
                    tab[sch][c] = float(fn(xs)) if len(xs) == len(RUNS) else None
            res.setdefault(stat, {})[k] = tab
    res["median"]["table_columns"] = cols
    res["margin_05"] = res["runs"].pop("margin_05")
    res["margin_05_always_fail_views"] = res["always_fail_views"].pop("margin_05")
    res["pooled_feasibility"] = pooled_tables(a)
    with open(os.path.join(a.out, "rev6_cold4_r5_runs.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    for th in (30,):
        k = f"table_cold_{th}dB"
        print("columns", cols)
        for sch in res["median"][k]:
            print(f"{sch:24s}", "  ".join(f"{res['median'][k][sch][c]:.2e} [{res['min'][k][sch][c]:.2e},{res['max'][k][sch][c]:.2e}]"
                                          if res["median"][k][sch][c] is not None else "None" for c in cols))
        pf = res["pooled_feasibility"]
        print("pooled feasibility: median [min, max] over run1-run3; margin 0.5")
        for sch in pf["median"][k]:
            print(f"{sch:24s}", "  ".join(f"{pf['median'][k][sch][c]:.2e} [{pf['min'][k][sch][c]:.2e},{pf['max'][k][sch][c]:.2e}] m05 {pf['runs']['margin_05'][k][sch][c]:.2e}"
                                          for c in cols))
        print("substituted", pf["substituted_rows"], "borderline", json.dumps(pf["borderline_views"]))
        print("margin 0.5")
        for sch in res["margin_05"][k]:
            print(f"{sch:24s}", "  ".join(f"{res['margin_05'][k][sch][c]:.2e}" if res["margin_05"][k][sch][c] is not None else "None" for c in cols))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--tag", default="_v2m_r1")
    main(ap.parse_args())
