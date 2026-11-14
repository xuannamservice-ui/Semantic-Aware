"""Generate the LaTeX tables of the evaluation section from the result files (no hand transcription).
usage: python gen_tables.py <results root containing results/ and hac/> <splitstep dir> <out dir>
"""
import glob
import json
import os
import sys
from math import comb

import numpy as np

ROOT, SS, OUT = sys.argv[1:4]
R = os.path.join(ROOT, "results")
SCENES = [("Mip-NeRF 360", ["bicycle", "garden", "stump", "room", "counter", "kitchen", "bonsai"]),
          ("Tanks and Temples", ["truck", "train"]), ("Deep Blending", ["playroom", "drjohnson"])]
ALL = [s for _, ss in SCENES for s in ss]
POL = "interleaved"
rng = np.random.default_rng(0)


def J(p):
    return json.load(open(p)) if os.path.exists(p) else None


def mb(m):
    return sum(m["scene_bytes"].values()) / 1e6


def boot(x, n=4000):
    x = np.asarray(x); i = rng.integers(0, len(x), (n, len(x))); m = x[i].mean(1)
    return np.percentile(m, [2.5, 97.5])


def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def rows_of(e, pol=POL):
    by = {}
    for r in e["rows"]:
        if r["policy"] == pol:
            by.setdefault(r["method"], []).append(r)
    return by


def w(name, text):
    open(os.path.join(OUT, name), "w").write(text)
    print(f"--- {name}\n{text}")


facts = {}

# ---------------- reference 3DGS table ----------------
lines = []
for ds, ss in SCENES:
    for s in ss:
        g = J(f"{R}/{s}/metrics_3dgs.json")
        img = {"Mip-NeRF 360": "images\\_4" if s in ("bicycle", "garden", "stump") else "images\\_2"}.get(ds, "images")
        lines.append(f"{s} & {ds}, {img} & {g['test']['psnr']:.2f} & {g['test']['ssim']:.3f} & {g['test']['lpips']:.3f} \\\\")
w("tab_ref3dgs.tex", "\n".join(lines))

# ---------------- main per-scene table ----------------
lines, acc = [], {k: [] for k in ("g", "p", "l", "mb", "drop", "idw", "gen", "tern", "psi", "sep")}
for ds, ss in SCENES:
    for j, s in enumerate(ss):
        g = J(f"{R}/{s}/metrics_3dgs.json")["test"]; m = J(f"{R}/{s}/metrics_layered_v2a_r1.json")
        e = J(f"{R}/{s}/eval_layered_v2a_r1_T8R8_2.5.json")
        if e is None:
            lead = ("\\multirow{%d}{*}{%s}" % (len(ss), ds)) if j == 0 else ""
            lines.append(lead + " & " + s + " & MISSING \\\\"); continue
        e = e["summary"]
        sp = J(f"{R}/{s}/eval_layered_v2s_r1_T8R8_2.5.json")
        v = dict(g=g["psnr"], p=m["test_clean"]["psnr"], l=m["test_clean"]["lpips"], mb=mb(m),
                 drop=e[f"{POL}/drop"]["psnr"], idw=e[f"{POL}/idw"]["psnr"], gen=e[f"{POL}/gen_fp"]["psnr"],
                 tern=e[f"{POL}/gen_tern"]["psnr"], psi=e[f"{POL}/gen_fp"]["psnr_vs_full"])
        for k in v:
            acc[k].append(v[k])
        sep = f"{sp['summary'][POL + '/gen_fp']['psnr']:.2f}" if sp else "n/e"
        if sp:
            acc["sep"].append((s, sp["summary"][POL + "/gen_fp"]["psnr"], v["gen"]))
        lead = f"\\multirow{{{len(ss)}}}{{*}}{{{ds}}}" if j == 0 else ""
        lines.append(f"{lead} & {s} & {v['g']:.2f} & {v['p']:.2f} & {v['l']:.3f} & {v['mb']:.2f} & {v['drop']:.2f} & "
                     f"{v['idw']:.2f} & {v['gen']:.2f} & {v['tern']:.2f} & {v['psi']:.1f} & {sep} \\\\")
    lines.append("\\hline")
mean = {k: np.mean(v) for k, v in acc.items() if k != "sep"}
lines.append(f"\\multicolumn{{2}}{{|l|}}{{Mean over the 11 scenes}} & {mean['g']:.2f} & {mean['p']:.2f} & {mean['l']:.3f} & "
             f"{mean['mb']:.2f} & {mean['drop']:.2f} & {mean['idw']:.2f} & {mean['gen']:.2f} & {mean['tern']:.2f} & "
             f"{mean['psi']:.1f} & -- \\\\")
w("tab_main.tex", "\n".join(lines))
facts["main_mean"] = mean
facts["sep_pairs"] = acc["sep"]
facts["clean_minus_3dgs"] = [p - g for p, g in zip(acc["p"], acc["g"])]

# ---------------- pooled statistics ----------------
for cfg in ("T8R8", "T6R11"):
    d, gap, gapd, gapi, n = [], [], [], [], 0
    for s in ALL:
        e = J(f"{R}/{s}/eval_layered_v2a_r1_{cfg}_2.5.json")
        if e is None:
            print("missing", s, cfg); continue
        by = rows_of(e)
        d += [a["psnr"] - b["psnr"] for a, b in zip(by["gen_fp"], by["idw"])]
        gap.append(np.mean([r["full_psnr"] - r["psnr"] for r in by["gen_fp"]]))
        gapd.append(np.mean([r["full_psnr"] - r["psnr"] for r in by["drop"]]))
        gapi.append(np.mean([r["full_psnr"] - r["psnr"] for r in by["idw"]]))
    lo, hi = boot(d)
    facts[f"pooled_{cfg}"] = dict(gen_minus_idw=float(np.mean(d)), ci=[float(lo), float(hi)], win=float(np.mean(np.array(d) > 0)),
                                  n=len(d), gap_gen_range=[float(min(gap)), float(max(gap))], gap_gen_mean=float(np.mean(gap)),
                                  gap_drop_mean=float(np.mean(gapd)), gap_idw_mean=float(np.mean(gapi)))

# by K
accK = {}
for s in ALL:
    for r in (J(f"{R}/{s}/eval_layered_v2a_r1_T8R8_2.5.json") or {"rows": []})["rows"]:
        if r["policy"] == POL:
            accK.setdefault((r["K"], r["method"]), []).append(r["full_psnr"] - r["psnr"])
lines = []
for K in sorted({k for k, _ in accK}):
    if len(accK[(K, "drop")]) < 20:
        continue
    lines.append(f"{K} & {len(accK[(K, 'drop')])} & {np.mean(accK[(K, 'drop')]):.2f} & {np.mean(accK[(K, 'base')]):.2f} & "
                 f"{np.mean(accK[(K, 'idw')]):.2f} & {np.mean(accK[(K, 'gen_fp')]):.2f} & {np.mean(accK[(K, 'gen_tern')]):.2f} \\\\")
w("tab_byK.tex", "\n".join(lines))

# policies: mean over scenes
lines = []
for pol in ("random", "contiguous", "interleaved", "importance"):
    cells = []
    for cfg in ("T6R11", "T8R8"):
        ss = [x["summary"] for x in (J(f"{R}/{s}/eval_layered_v2a_r1_{cfg}_2.5.json") for s in ALL) if x]
        cells += [np.mean([x[f"{pol}/{m}"]["psnr"] for x in ss]) for m in ("drop", "idw", "gen_fp")]
        cells += [np.mean([x[f"{pol}/gen_fp"]["psnr_vs_full"] for x in ss])]
    lines.append(f"{pol.capitalize()} & " + " & ".join(f"{c:.2f}" if i % 4 != 3 else f"{c:.1f}" for i, c in enumerate(cells)) + " \\\\")
w("tab_policy.tex", "\n".join(lines))

# garden detail (interleaved) incl. variants
lines = []
for tag, name in (("_v2a_r1", "Joint, $\\lambda=1$, $\\lambda_{rate}=5\\times10^{-5}$"), ("_v2a_r0", "Joint, $\\lambda=1$, $\\lambda_{rate}=0$"),
                  ("_v2a_r2", "Joint, $\\lambda=1$, $\\lambda_{rate}=2\\times10^{-4}$"), ("_v2a_r1_l03", "Joint, $\\lambda=0.3$, $\\lambda_{rate}=5\\times10^{-5}$"),
                  ("_v2s_r1", "Separate, $\\lambda_{rate}=5\\times10^{-5}$"), ("_v2s_r0", "Separate, $\\lambda_{rate}=0$")):
    m = J(f"{R}/garden/metrics_layered{tag}.json")
    if not m:
        continue
    for cfg in ("T6R11", "T8R8"):
        e = J(f"{R}/garden/eval_layered{tag}_{cfg}_2.5.json")
        if not e:
            continue
        s = e["summary"]
        lines.append(f"{name} & {mb(m):.2f} & {cfg} & {m['test_clean']['psnr']:.2f} & {s[POL + '/drop']['psnr']:.2f} & {s[POL + '/base']['psnr']:.2f} & "
                     f"{s[POL + '/idw']['psnr']:.2f} & {s[POL + '/gen_fp']['psnr']:.2f} & {s[POL + '/gen_fp']['lpips']:.3f} & "
                     f"{s[POL + '/gen_tern']['psnr']:.2f} & {s[POL + '/gen_fp']['psnr_vs_full']:.1f} & {s[POL + '/drop']['lost_frac']:.2f} \\\\")
w("tab_codec_garden.tex", "\n".join(lines))

# edge variant
ed = []
for s in ("garden", "stump"):
    for cfg in ("T8R8", "T6R11"):
        e = J(f"{R}/{s}/eval_layered_v2a_r1_edge_{cfg}_2.5.json")
        if e:
            ed.append((s, cfg, e["summary"][POL + "/gen_fp"]["psnr"], e["summary"][POL + "/edge_fp"]["psnr"], e["summary"][POL + "/edge_tern"]["psnr"]))
facts["edge"] = ed

# ---------------- split-step: link and codec ----------------
lines = []
for s2 in ("1.0", "2.5"):
    res = J(os.path.join(SS, f"results_{s2}.json"))
    facts[f"ss_{s2}_keys"] = list(res.keys())[:20]
w("ss_keys.txt", json.dumps({k: facts[k] for k in facts if k.startswith("ss_")}, indent=1))
lines = []
for cfg in ("T8R8", "T6R8"):
    for s in ALL:
        e = J(f"{R}/{s}/eval_layered_v2a_r1_{cfg}_ss2.5.json")
        z = J(f"{R}/{s}/eval_layered_v2a_r1_{cfg}_2.5.json")
        if not e:
            continue
        su = e["summary"]
        zs = f"{z['summary'][POL + '/gen_fp']['psnr']:.2f}" if z else "n/e"
        lines.append(f"{cfg} & {s} & {su['full']['psnr']:.2f} & {su[POL + '/drop']['lost_frac']:.2f} & {su[POL + '/drop']['psnr']:.2f} & "
                     f"{su[POL + '/idw']['psnr']:.2f} & {su[POL + '/gen_fp']['psnr']:.2f} & {zs} \\\\")
w("tab_ss_codec.tex", "\n".join(lines))

# ---------------- frames ----------------
for tau in ("10", "2.5"):
    D, meta = {}, {}
    for s in ALL:
        fr = J(f"{R}/{s}/frames_v2a_r1_T8R8_tau{tau}_eps0.166667.json")
        if not fr:
            continue
        for r in fr["rows"]:
            if r["chain"] >= 10:
                continue
            D.setdefault((s, r["chain"], r["view"], r["frame"]), {})[r["scheme"]] = r
    keys = list(D)
    sch = ["none", "conceal", "arq", "arq+conceal", "mds", "mds+conceal"]
    tab = {}
    for sc in sch:
        psi = np.array([min(D[k][sc]["psi"], 99) for k in keys])
        tab[sc] = dict(n=len(keys), miss=int(sum(D[k][sc]["n_missing"] > 0 for k in keys)), f30=int((psi < 30).sum()),
                       f35=int((psi < 35).sum()), psnr=float(np.mean([D[k][sc]["psnr"] for k in keys])), psimin=float(psi.min()))
    tests = {}
    for th in (30, 35):
        fail = {sc: {k for k in keys if D[k][sc]["psi"] < th} for sc in sch}
        for a, b in (("conceal", "none"), ("conceal", "arq"), ("conceal", "mds"), ("arq+conceal", "arq"), ("mds+conceal", "mds"), ("arq+conceal", "conceal")):
            tests[f"{a}|{b}|{th}"] = dict(b_only=len(fail[b] - fail[a]), a_only=len(fail[a] - fail[b]),
                                          p=mcnemar(len(fail[b] - fail[a]), len(fail[a] - fail[b])))
    facts[f"frames_T8R8_tau{tau}"] = dict(table=tab, tests=tests)
other = {}
for cfg in ("T6R8", "T6R11", "T8R11"):
    n = miss = f30 = f35 = 0; nsc = 0
    for s in ALL:
        fr = J(f"{R}/{s}/frames_v2a_r1_{cfg}_tau10_eps0.166667.json")
        if not fr:
            continue
        nsc += 1; su = fr["summary"]["none"]; n += su["frames"]; miss += su["frames_with_missing"]
        f30 += round(su["p_fail_psi30.0"] * su["frames"]); f35 += round(su["p_fail_psi35.0"] * su["frames"])
    other[cfg] = dict(scenes=nsc, frames=n, frames_with_missing=miss, f30=f30, f35=f35)
facts["frames_other"] = other
stride = {}
for f in glob.glob(f"{R}/*/frames_v2a_r1_T8R8_tau10_eps0.166667_stride*.json"):
    fr = J(f); stride.setdefault(fr["stride"], {})[fr["scene"]] = {k: (v["frames"], v["frames_with_missing"], round(v["p_fail_psi30.0"] * v["frames"]),
                                                                     round(v["p_fail_psi35.0"] * v["frames"])) for k, v in fr["summary"].items()}
facts["stride"] = stride

# ---------------- HAC ----------------
hac = {}
for f in glob.glob(os.path.join(ROOT, "hac/logs/*_*.results.json")):
    b = os.path.basename(f).replace(".results.json", "")
    s, lam = b.split("_")
    r = J(f)["ours_30000"]
    log = open(f.replace(".results.json", ".stdout.log")).read() if os.path.exists(f.replace(".results.json", ".stdout.log")) else ""
    enc = [l for l in log.splitlines() if l.startswith("Encoded sizes")]
    size = float(enc[-1].split("Total")[1].split(",")[0]) if enc else None
    hac[b] = dict(psnr=r["PSNR"], ssim=r["SSIM"], lpips=r["LPIPS"], size_MB=size)
for f in glob.glob(os.path.join(ROOT, "hac/trace_eval/*.json")):
    b = os.path.basename(f).replace(".json", "")
    hac["trace_" + b] = {k: v for k, v in J(f)["summary"].items()}
facts["hac"] = hac
json.dump(facts, open(os.path.join(OUT, "facts.json"), "w"), indent=1, default=float)
print(json.dumps({k: facts[k] for k in ("main_mean", "pooled_T8R8", "pooled_T6R11", "frames_other", "edge")}, indent=1, default=float))
