"""Extrapolation of the outage to an infinite Zernike expansion: the outage of results/jconv_<s2>.json for J = 435, 595,
780 (64 x 64 grid, paired samples) is fitted linearly in the Noll residual phase variance Delta_J and evaluated at
Delta_J = 0. usage: python jconv_extrap.py 2.5"""
import json, os, sys
import numpy as np
import oamfso as of

s2 = sys.argv[1] if len(sys.argv) > 1 else "2.5"
d = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", f"jconv_{s2}.json")))
Js = (435, 595, 780); x = np.array([of.noll_residual(J) for J in Js])
for k, c in d["configs"].items():
    y = np.array([c[f"64_{J}"]["outage"] for J in Js])
    A = np.vstack([np.ones(3), x]).T; co = np.linalg.lstsq(A, y, rcond=None)[0]
    print(f"{k}: outage {y} -> J->inf {co[0]:.4g} (x{co[0] / y[0]:.3f} of J=435), max fit residual {np.abs(y - A @ co).max():.1e}")
