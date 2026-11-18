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
           the same view from the other runs (same windows); per run, then median / min / max over run1-run3;
           adaptive = MDS + conceal on the views where the code is feasible, concealment alone (its own result for the
           view) where it is not; window_median: the table from one outcome per window, the execution with the median
           total time over run1-run3 (window_median())
  protocol_stats: per column, the statistics quoted in the text under the same protocol (undeliverable views, views
           whose redundant schedule is infeasible, windows per scheme, deadline misses, window classes of concealment
           alone, one-sided 95% bound of a zero-failure entry as fill.py computes it)
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


TF = 11.1
_POOL = {}


def load_pool(a):
    """Per-scene one-frame files of all runs and the pooled feasibility of every view and schedule."""
    if not _POOL:
        sfxs = {**RUNS, "margin_05": "_m05"}
        J = {(r, cfg, tau, s): json.load(open(os.path.join(a.out, s, f"cold4{a.tag}_{cfg}_tau{tau}_r5{sx}.json")))
             for r, sx in sfxs.items() for cfg, tau in CFGS for s in ALLS
             if os.path.exists(os.path.join(a.out, s, f"cold4{a.tag}_{cfg}_tau{tau}_r5{sx}.json"))}
        poolA, poolR = {}, {}
        for (r, cfg, tau, s), j in J.items():
            for v in j["views"]:
                poolA.setdefault((s, v["view"]), []).extend(v["free_A"]["totals"])
                poolR.setdefault((s, cfg, v["view"]), []).extend(v["free_R"]["totals"])
        infA = {k: float(np.median(x)) > TF for k, x in poolA.items()}
        infR = {k: float(np.median(x)) > TF for k, x in poolR.items()}
        _POOL.update(sfxs=sfxs, J=J, poolA=poolA, poolR=poolR, infA=infA, infR=infR)
    p = _POOL
    return p["sfxs"], p["J"], p["poolA"], p["poolR"], p["infA"], p["infR"]


def infeasible(infA, infR, s, cfg, view, sch):
    return infR[(s, cfg, view)] if sch.startswith(("rep2", "mds")) else infA[(s, view)]


def window_median(a, cfg, tau, s):
    """One outcome per window (view, stratum, window, scheme) from the margin-0 runs run1-run3: the execution whose total
    time is the median of the executions of that window (upper median if two; a run in which the view was infeasible
    for the scheme has no execution of it); a window whose anchor payload is lost in every run is 'anchor_lost', and a
    view that is infeasible for the scheme under the pooled rule is 'always_fail' (failed) whatever a run executed."""
    _, J, _, _, infA, infR = load_pool(a)
    W = {}
    for r in RUNS:
        j = J.get((r, cfg, tau, s))
        if j is None:
            continue
        for x in j["rows"]:
            W.setdefault((x["view"], x["stratum"], x["win"], x["scheme"]), []).append(x)
    out = {}
    for k, xs in W.items():
        view, stratum, win, sch = k
        if infeasible(infA, infR, s, cfg, view, sch):
            out[k] = {"variant": "always_fail", "psi": 0.0, "executed": False, "deadline_miss": True, "n_exec": 0}
            continue
        ex = sorted([x for x in xs if x.get("total_ms") is not None], key=lambda x: x["total_ms"])
        if ex:
            m = ex[len(ex) // 2]
            out[k] = {"variant": m["variant"], "psi": m["psi"], "executed": True, "deadline_miss": bool(m["deadline_miss"]), "n_exec": len(ex),
                      "total_ms": m["total_ms"]}
        elif any(x["variant"] == "anchor_lost" for x in xs):
            out[k] = {"variant": "anchor_lost", "psi": 0.0, "executed": False, "deadline_miss": False, "n_exec": 0}
        else:
            out[k] = {"variant": "missing", "psi": 0.0, "executed": False, "deadline_miss": False, "n_exec": 0}
    return out


def window_median_tables(a):
    """Table from the window medians: per view p_stratum x share of failed window medians (1 if infeasible under the
    pooled rule), adaptive = MDS + conceal where the code is feasible, concealment alone where it is not; undeliverable
    views left out; mean over the views of a scene, then over the scenes."""
    sfxs, J, _, _, infA, infR = load_pool(a)
    tabs = {th: {} for th in (30, 35)}
    for cfg, tau in CFGS:
        col = f"{cfg}|tau{tau}|_r5"; per = {}
        for s in ALLS:
            j = J.get(("run1", cfg, tau, s))
            if j is None:
                continue
            wm = window_median(a, cfg, tau, s)
            vs = [v for v in j["views"] if not infA[(s, v["view"])]]
            for th in (30, 35):
                pv = {}
                for sch in ec4.SCHEMES:
                    kind = "D" if sch.startswith("mds") else "E"
                    for v in vs:
                        if infeasible(infA, infR, s, cfg, v["view"], sch):
                            pv[(sch, v["view"])] = 1.0; continue
                        ws = [w for k, w in wm.items() if k[0] == v["view"] and k[1] == kind and k[3] == sch]
                        pv[(sch, v["view"])] = (v["pD"] if kind == "D" else v["pE"]) * (float(np.mean([w["psi"] < th for w in ws])) if ws else 0.0)
                for sch in ec4.SCHEMES:
                    per.setdefault((sch, th), []).append(float(np.mean([pv[(sch, v["view"])] for v in vs])))
                per.setdefault(("adaptive", th), []).append(float(np.mean(
                    [pv[("conceal", v["view"])] if infR[(s, cfg, v["view"])] else pv[("mds_conceal", v["view"])] for v in vs])))
        for th in (30, 35):
            for sch in ec4.SCHEMES[:6] + ("adaptive",) + ec4.SCHEMES[6:]:
                tabs[th].setdefault(sch, {})[col] = float(np.mean(per[(sch, th)]))
    return {f"table_cold_{th}dB": tabs[th] for th in (30, 35)}


def protocol_stats(a):
    """Statistics of the text under the protocol of the table (pooled feasibility, window medians of run1-run3)."""
    sfxs, J, poolA, poolR, infA, infR = load_pool(a)
    out = {"definitions": {
        "undeliverable": "view whose loss-free plain schedule (S_A slots) has a pooled median total > 11.1 ms (all loss-free executions "
                         "of the view: 4 runs x 3 executions x 2 tau_c x 2 configurations; the schedule does not depend on the configuration)",
        "infeasible_for_scheme": "deliverable view whose loss-free replication/MDS schedule (S_R slots) has a pooled median total > 11.1 ms "
                                 "(4 runs x 3 executions x 2 tau_c of that configuration); counts as failed (p = 1) for rep2, rep2_conceal, mds, mds_conceal",
        "window_median": window_median.__doc__.strip(),
        "zero_bound": "fill.py cold_zero_bound: mean over the scenes of the mean over the deliverable views of p_E (p_D for MDS) x 3 / "
                      "number of windows of the scheme on the deliverable views"}}
    for cfg, tau in CFGS:
        col = f"{cfg}|tau{tau}|_r5"; st = {}
        und = {s: sorted(v["view"] for v in J[("run1", cfg, tau, s)]["views"] if infA[(s, v["view"])]) for s in ALLS}
        nv = sum(len(J[("run1", cfg, tau, s)]["views"]) for s in ALLS); nu = sum(len(x) for x in und.values())
        st["undeliverable"] = {"n": nu, "of": nv, "pct": 100.0 * nu / nv, "per_scene": {s: len(x) for s, x in und.items() if x},
                               "views": {s: x for s, x in und.items() if x}}
        # views whose redundant schedule is infeasible (the replication / MDS entries) and their weight in the column
        ndel = {s: sum(not infA[(s, v["view"])] for v in J[("run1", cfg, tau, s)]["views"]) for s in ALLS}
        infv = [{"scene": s, "view": v["view"], "S_A": v["S_A"], "S_R": v["S_R"], "n_visible": v["n_visible"],
                 "pooled_median_R_ms": float(np.median(poolR[(s, cfg, v["view"])])), "pooled_median_A_ms": float(np.median(poolA[(s, v["view"])])),
                 "frac_R_executions_above": float(np.mean(np.asarray(poolR[(s, cfg, v["view"])]) > TF)), "weight": 1.0 / (ndel[s] * len(ALLS))}
                for s in ALLS for v in J[("run1", cfg, tau, s)]["views"] if not infA[(s, v["view"])] and infR[(s, cfg, v["view"])]]
        st["redundancy_infeasible_views"] = {"n": len(infv), "sum_of_weights": float(sum(x["weight"] for x in infv)), "views": infv,
                                             "deliverable_views_per_scene": ndel}
        # windows per scheme on the deliverable views, from the window medians
        WM = {s: window_median(a, cfg, tau, s) for s in ALLS}
        win = {}
        for sch in ec4.SCHEMES:
            kind = "D" if sch.startswith("mds") else "E"
            ws = [w for s in ALLS for k, w in WM[s].items() if k[3] == sch and k[1] == kind and not infA[(s, k[0])]]
            c = {}
            for w in ws:
                c[w["variant"]] = c.get(w["variant"], 0) + 1
            win[sch] = {"stratum": kind, "windows": len(ws), "executed": sum(w["executed"] for w in ws), "anchor_lost": c.get("anchor_lost", 0),
                        "infeasible": c.get("always_fail", 0), "classes": c, "below30": sum(w["psi"] < 30 for w in ws),
                        "deadline_miss_executed": sum(w["executed"] and w["deadline_miss"] for w in ws),
                        "n_exec_per_window": {str(n): sum(w["n_exec"] == n for w in ws if w["executed"]) for n in (1, 2, 3)}}
        st["windows_per_scheme"] = win
        exw = [w for s in ALLS for k, w in WM[s].items() if w["executed"] and not infA[(s, k[0])]]
        nm = sum(w["deadline_miss"] for w in exw)
        st["deadline_miss_window_median"] = {"n_miss": nm, "n_executed": len(exw), "frac": nm / max(1, len(exw)),
                                             "cp95_one_sided_upper": ec4.cp_upper(nm, len(exw)) if exw else None}
        runs_ = {}
        for r in RUNS:
            ex = [x for s in ALLS for x in J[(r, cfg, tau, s)]["rows"] if x.get("total_ms") is not None and not infA[(s, x["view"])]
                  and not infeasible(infA, infR, s, cfg, x["view"], x["scheme"])]
            m_ = sum(bool(x["deadline_miss"]) for x in ex)
            runs_[r] = {"n_miss": m_, "n_executed": len(ex), "frac": m_ / max(1, len(ex))}
        st["deadline_miss_per_run"] = runs_
        st["deadline_miss_per_run_range_pct"] = [100 * min(x["frac"] for x in runs_.values()), 100 * max(x["frac"] for x in runs_.values())]
        # classes of the windows of concealment alone in which some stream fails (stratum E), all 338 views
        cls = {}
        for s in ALLS:
            for k, w in WM[s].items():
                if k[3] != "conceal" or k[1] != "E":
                    continue
                c_ = "undeliverable" if infA[(s, k[0])] else w["variant"]
                cls.setdefault(c_, [0, 0]); cls[c_][0] += 1; cls[c_][1] += int(w["psi"] < 30)
        st["conceal_window_classes"] = {"n_windows": sum(x[0] for x in cls.values()), "n_below30": sum(x[1] for x in cls.values()),
                                        "classes": {c_: {"windows": x[0], "below30": x[1]} for c_, x in cls.items()},
                                        "legend": {"complete": "view arrived complete", "full": "network conceals every lost anchor",
                                                   "anytime": "network on part, IDW on the rest", "idw": "IDW on all lost anchors",
                                                   "drop": "rendered from what arrived", "anchor_lost": "anchor payload lost",
                                                   "undeliverable": "view undeliverable within one frame (pooled rule)"}}
        # one-sided 95% bound for a zero-failure entry (fill.py cold_zero_bound), one frame, deliverable views
        zb = {}
        for kind, key in (("E", "pE"), ("D", "pD")):
            pk = float(np.mean([np.mean([v[key] for v in J[("run1", cfg, tau, s)]["views"] if not infA[(s, v["view"])]]) for s in ALLS]))
            n_ = win["conceal" if kind == "E" else "mds_conceal"]["windows"]
            zb[kind] = {"mean_p_stratum": pk, "n_windows": n_, "bound": pk * 3.0 / max(1, n_)}
        st["zero_failure_bound_one_frame"] = zb
        # the same bound within two frames (run1 _r6 files, as fill.py; no view is undeliverable there)
        f6 = [os.path.join(a.out, s, f"cold4{a.tag}_{cfg}_tau{tau}_r6.json") for s in ALLS]
        if all(os.path.exists(f) for f in f6):
            j6 = [json.load(open(f)) for f in f6]; zb6 = {}
            for kind, key in (("E", "pE"), ("D", "pD")):
                vs6 = [[v for v in j["views"] if not v["always_fail"]["none"]] for j in j6]
                pk = float(np.mean([np.mean([v[key] for v in vs]) for vs in vs6]))
                ok = [{v["view"] for v in vs} for vs in vs6]
                n_ = sum(1 for j, o in zip(j6, ok) for x in j["rows"] if x["view"] in o and x["stratum"] == kind
                         and x["scheme"] == ("conceal" if kind == "E" else "mds_conceal"))
                zb6[kind] = {"mean_p_stratum": pk, "n_windows": n_, "bound": pk * 3.0 / max(1, n_)}
            st["zero_failure_bound_two_frames_run1"] = zb6
        out[col] = st
    return out


def pooled_tables(a):
    """Tables with the feasibility of each view decided from the pooled loss-free executions (see the module doc)."""
    sfxs, J, poolA, poolR, infA, infR = load_pool(a)
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
    res["pooled_feasibility"]["window_median"] = window_median_tables(a)
    res["protocol_stats"] = protocol_stats(a)
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
