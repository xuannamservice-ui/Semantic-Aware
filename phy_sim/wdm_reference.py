"""WDM reference for Table 4: M wavelength channels on one Gaussian beam (no inter-channel crosstalk).

Same link budget per channel as the OAM streams (run_modeset.snr0(M): total power split over M channels, same
demultiplexer insertion loss and receiver noise). All wavelengths cross the same path, so their irradiance
fluctuation is taken as common: one aperture-averaged Gamma-Gamma factor I per slot (oamfso.gg_aperture,
receive aperture D = 2 w(L)); a large-area photodiode collects the power of the Gaussian beam inside the aperture,
eta = 1 - exp(-2 a^2 / w(L)^2) (no mode matching, so no crosstalk and no phase-induced coupling loss). A channel fails
if snr0(M) * (eta I)^2 < gamma_th (the IM/DD electrical SNR scales with the square of the received power, as in the
OAM model). Consequently the M channels fail together: the outage is all-or-nothing.
Monte Carlo with 1e7 draws per point (seed 9911) next to the exact Gamma-Gamma CDF.
"""
import json
import numpy as np
import oamfso as of
import run_modeset as rm

rng = np.random.default_rng(9911)
w0, zR, wL = of.beam_params()
eta = 1 - np.exp(-2 * of.A_R ** 2 / wL ** 2)
print(f"w(L) = {wL*100:.2f} cm, aperture radius {of.A_R*100:.1f} cm, captured fraction eta = {eta:.4f}")
out = {"eta": float(eta), "w_L_m": float(wL)}
for s2 in (1.0, 2.5):
    a, b = of.gg_aperture(s2)
    X = rng.gamma(a, 1 / a, 10_000_000) * rng.gamma(b, 1 / b, 10_000_000)
    for M in (4, 6, 8):
        snr = rm.snr0(M)
        thr = np.sqrt(rm.GTH / snr) / eta             # I below thr -> SNR below gamma_th
        p_mc = float((X < thr).mean())
        p_cdf = float(of.gg_cdf_I(thr, a, b)) if hasattr(of, "gg_cdf_I") else None
        out[f"WDM{M}_{s2}"] = {"M": M, "s2": s2, "alpha": a, "beta": b, "snr0_dB": 10 * np.log10(snr),
                               "I_threshold": thr, "outage_mc": p_mc, "outage_cdf": p_cdf,
                               "streams_lost_in_outage": M}
        print(f"s2={s2} WDM M={M}: snr0 {10*np.log10(snr):.1f} dB, I_thr {thr:.4f}, outage MC {p_mc:.3e} "
              f"(CDF {p_cdf if p_cdf is None else f'{p_cdf:.3e}'}), all {M} channels lost together")
json.dump(out, open("results/wdm_reference.json", "w"), indent=1)
