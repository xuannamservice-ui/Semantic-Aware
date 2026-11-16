"""HAC with an MDS code at the transmitted bits of the layered codec (T8R8, four anchor copies).

The layered codec sends its representation plus three further copies of positions and base layer. The same budget lets
HAC protect its bitstream with an (8, 8 - r) MDS code over the eight streams, r the largest number of parity streams
with HAC_size * 8 / (8 - r) <= budget; the code recovers every slot with at most r failed streams, and the other slots
keep the quality of HAC with IDW of the lost anchors (hac/results/trace_eval). The layered codec can spend the same budget
on a code too: two anchor copies instead of four and an (8, 8 - r') MDS code on its enhancement layer; a slot then fails
if both anchor copies are lost (2.03e-3 per slot for T8R8, Table 4 of the paper) or if more than r' streams fail and the
concealed slot falls below the threshold. Share of outage slots below 30 and 35 dB on the same failure patterns
(interleaved mapping).
usage: python hac_eqbits.py <codec/results> <hac/results> <out.json>
"""
import glob
import json
import os
import sys

import numpy as np

COD, HAC, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
res = {}
for f in sorted(glob.glob(os.path.join(HAC, "trace_eval", "*_0.004_T8R8.json"))):
    s = os.path.basename(f).split("_")[0]
    enc = [l for l in open(os.path.join(HAC, "logs", f"{s}_0.004.stdout.log")).read().splitlines() if l.startswith("Encoded sizes")]
    hac_mb = float(enc[-1].split("Total")[1].split(",")[0])
    m = json.load(open(os.path.join(COD, s, "metrics_layered_v2a_r1.json"))); sb = m["scene_bytes"]
    budget = sum(sb.values()) / 1e6 + 3 * (sb["positions"] + sb["base"]) / 1e6
    r = max(q for q in range(8) if hac_mb * 8 / (8 - q) <= budget)
    hr = [x for x in json.load(open(f))["rows"] if x["policy"] == "interleaved" and x["method"] == "idw"]
    orw = [x for x in json.load(open(os.path.join(COD, s, "eval_layered_v2a_r1_T8R8_2.5.json")))["rows"]
           if x["policy"] == "interleaved" and x["method"] == "gen_fp"]
    size = sum(sb.values()) / 1e6; cp = (sb["positions"] + sb["base"]) / 1e6; enh = sb["enhancement"] / 1e6
    rl = max(q for q in range(8) if size + cp + enh * q / (8 - q) <= budget)
    res[s] = {"layered_total_MB": budget, "hac_MB": hac_mb, "parity_streams": r, "hac_with_code_MB": hac_mb * 8 / (8 - r),
              "layered_mds_parity_streams": rl, "layered_mds_MB": size + cp + enh * rl / (8 - rl)}
    for th in (30, 35):
        res[s][f"hac_mds_below_{th}"] = float(np.mean([x["K"] > r and x["psnr_vs_full"] < th for x in hr]))
        res[s][f"layered_below_{th}"] = float(np.mean([x["psnr_vs_full"] < th for x in orw]))
        res[s][f"layered_mds_below_{th}"] = float(2.03e-3 / 0.1455678 + np.mean([x["K"] > rl and x["psnr_vs_full"] < th for x in orw]))
    print(s, {k: round(v, 3) if isinstance(v, float) else v for k, v in res[s].items()})
json.dump(res, open(OUT, "w"), indent=1)
