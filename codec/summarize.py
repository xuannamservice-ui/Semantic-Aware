"""Print a compact comparison of all layered evaluations (policy 'importance' unless --policy)."""
import glob, json, sys
pol = sys.argv[1] if len(sys.argv) > 1 else "importance"
for fn in sorted(glob.glob("results/*/eval_layered_*_2.5.json")):
    d = json.load(open(fn)); S = d["summary"]; f = S["full"]
    g = lambda m, k="psnr": S[f"{pol}/{m}"][k]
    print(f"{d['scene']:8s} {d['cfg']:5s} {d['tag']:5s} full {f['psnr']:.2f}/{f['lpips']:.3f} | drop {g('drop'):.2f} "
          f"base {g('base'):.2f} idw {g('idw'):.2f} gen {g('gen_fp'):.2f}/{g('gen_fp','lpips'):.3f} tern {g('gen_tern'):.2f} "
          f"| gen-vs-full {g('gen_fp','psnr_vs_full'):.1f} dB | lost {g('gen_fp','lost_frac'):.2f}")
