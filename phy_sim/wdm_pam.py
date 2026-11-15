"""WDM reference with multilevel intensity modulation at the spectral efficiency of OAM wavelength reuse.

W carriers on one Gaussian beam (wdm_reference.py: common aperture-averaged Gamma-Gamma factor I, captured fraction
eta, no crosstalk), each with PAM-L. At equal average optical power the eye opening of PAM-L is 1/(L-1) of that of OOK,
so the decision SNR is snr0(W) (eta I)^2 / (L-1)^2 and a carrier fails if it falls below gamma_th (the OOK threshold;
the PAM-L symbol-error threshold differs only by the factor 2(L-1)/L in the error coefficient). Outage from the exact
Gamma-Gamma distribution function.
Reuse reference: 8 OOK streams on 4 carriers = 2 bit/symbol per carrier = WDM with 4 carriers and PAM-4.
Output: results/wdm_pam.json
"""
import json
import math
import numpy as np
import oamfso as of
import run_modeset as rm

w0, zR, wL = of.beam_params()
eta = 1 - math.exp(-2 * of.A_R ** 2 / wL ** 2)
out = {}
for s2 in (0.2, 0.5, 1.0, 2.5):
    a, b = of.gg_aperture(s2)
    for W in (4, 8):
        for L in (2, 4, 8):
            thr = math.sqrt(rm.GTH / rm.snr0(W)) * (L - 1) / eta
            p = float(of.gg_cdf_I(thr, a, b))
            out[f"W{W}_PAM{L}_{s2}"] = {"carriers": W, "levels": L, "bits_per_symbol_per_carrier": math.log2(L), "s2": s2,
                                        "snr0_dB": 10 * math.log10(rm.snr0(W)), "penalty_dB": 20 * math.log10(L - 1) if L > 2 else 0.0,
                                        "outage": p}
            print(f"s2={s2} W={W} PAM{L}: {math.log2(L):.0f} b/sym/carrier, outage {p:.2e}")
json.dump(out, open("results/wdm_pam.json", "w"), indent=1)
