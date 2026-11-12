"""Regression and consistency tests for oamfso.py (run: python test_oamfso.py <path to simB>)."""
import math, sys
import numpy as np
from scipy import integrate
import oamfso as of
sys.path.insert(0, sys.argv[1]); import phy_sim as ref     # independently cross-checked implementation

ok = True
def check(name, cond, info=""):
    global ok
    ok &= bool(cond); print(("PASS " if cond else "FAIL ") + name, info)

# 1) same model as the cross-checked reference (j = 2..20, independent coefficients, point receiver)
#    on the same random stream
ch = of.OAMChannel(j_min=2, j_max=20, point_receiver=True, correlated=False); rc = ref.OAMChannel()
for s2 in (0.2, 2.5):
    G1 = ch.draw(np.random.default_rng(5), 500, s2); G2 = rc.draw(np.random.default_rng(5), 500, s2)["G"]
    check(f"channel identical to reference (s2={s2})", np.allclose(G1, G2, rtol=1e-10, atol=0), f"max rel {np.max(np.abs(G1-G2)/np.abs(G2)):.1e}")
    for snr in (25, 68):
        P = 10 ** (snr / 10)
        a = of.sinr_mmse(G1, P); b = ref.sinr_mmse(G2, P)
        check(f"MMSE SINR equals reference explicit formula (s2={s2}, {snr} dB)", np.allclose(a, b, rtol=1e-8), f"max rel {np.max(np.abs(a-b)/b):.1e}")
        check(f"IaN SINR equals reference (s2={s2}, {snr} dB)", np.allclose(of.sinr_ian(G1, P), ref.sinr_ian(G2, P), rtol=1e-12))
# 2) Gamma-Gamma pdf, CDF and quantile
for (a, b) in ((4.035, 1.54), (7.33, 23.6), (47.4, 52.1)):
    tot = sum(integrate.quad(lambda x: of.gg_pdf_I(x, a, b), lo, hi, limit=200)[0] for lo, hi in ((0, 0.1), (0.1, 1), (1, 5), (5, np.inf)))
    check(f"pdf integrates to 1 (a={a}, b={b})", abs(tot - 1) < 1e-6, f"{tot:.8f}")
    x = 0.3
    F_quad = integrate.quad(lambda t: of.gg_pdf_I(t, a, b), 0, x, limit=200)[0]
    check(f"Meijer-G CDF equals quadrature (a={a}, b={b})", abs(of.gg_cdf_I(x, a, b) - F_quad) < 1e-8 * max(1, F_quad) + 1e-12, f"{of.gg_cdf_I(x,a,b):.6e} vs {F_quad:.6e}")
    for p in (1e-2, 1e-4):
        q = of.gg_quantile_I(p, a, b)
        check(f"quantile inverts CDF (p={p})", abs(of.gg_cdf_I(q, a, b) / p - 1) < 1e-6)
# 3) moment matching recovers the parameters of a Gamma-Gamma sample with the right ratio
rng = np.random.default_rng(9); a, b = 7.33, 23.6; n = 2_000_000
I = rng.gamma(a, 1 / a, n) * rng.gamma(b, 1 / b, n); gbar = 1e5
a2, b2, g2 = of.fit_equivalent_gg(gbar * I ** 2, a / b)
check("moment matching recovers (alpha, beta, gbar)", abs(a2 / a - 1) < 0.03 and abs(g2 / gbar - 1) < 0.01, f"{a2:.3f}, {b2:.3f}, {g2:.4g}")
# 3b) robust pdf and fast CDF agree with the Meijer-G CDF, including extreme parameters
for (a, b, x) in ((400.0, 0.95, 0.05), (1.2, 1.2, 1e-3), (300.0, 290.0, 0.6)):
    check(f"fast CDF equals Meijer-G (a={a}, b={b}, x={x})", abs(of.gg_cdf_fast(x, a, b) / of.gg_cdf_I(x, a, b) - 1) < 1e-6)
# 4) link budget and threshold
lb = of.link_budget()
i_p = 0.8 * 0.1 / 8 * 10 ** (-0.33); s_th = 4 * 1.380649e-23 * 300 * 10 ** 0.3 * 1e10 / 50; s_sh = 2 * 1.602176634e-19 * i_p * 1e10
check("link budget SNR0 (thermal with F_n = 3 dB plus shot noise)", abs(lb["snr0_dB"] - 10 * math.log10(i_p ** 2 / (s_th + s_sh))) < 1e-9,
      f"{lb['snr0_dB']:.3f} dB")
# 5) Noll covariance: diagonal, positive definiteness, sign of off-diagonal terms (Weber-Schafheitlin integral)
C = of.noll_cov(2, 153)
check("Noll covariance diagonal equals c_n", np.allclose(np.diag(C), [of.noll_cn(of.noll_nm(j)[0]) for j in range(2, 154)], rtol=1e-12))
check("Noll covariance positive definite up to j = 153", np.linalg.eigvalsh(C).min() > 0)
from scipy import special
def ws_num(n, n2, T=400.0):
    f = lambda t: special.jv(n + 1, t) * special.jv(n2 + 1, t) * t ** (-14 / 3)
    e = np.linspace(0, T, 801)
    return sum(integrate.quad(f, a, b, limit=200, epsabs=1e-16)[0] for a, b in zip(e[:-1], e[1:]))
for (j, j2) in ((4, 37), (4, 11), (7, 17)):        # (n, n') = (2, 8), (2, 4), (3, 5)
    n, n2 = of.noll_nm(j)[0], of.noll_nm(j2)[0]
    ratio = C[j - 2, j2 - 2] / C[j - 2, j - 2]
    g = special.gamma
    ws_diag = g(14 / 3) * g((2 * n + 2 - 14 / 3 + 1) / 2) / (2 ** (14 / 3) * g(17 / 6) ** 2 * g((2 * n + 2 + 14 / 3 + 1) / 2))
    num_ratio = (-1) ** ((n + n2 - 2 * of.noll_nm(j)[1]) // 2) * math.sqrt((n2 + 1) / (n + 1)) * ws_num(n, n2) / ws_diag
    check(f"Noll covariance C[{j},{j2}] matches the Bessel integral", abs(ratio / num_ratio - 1) < 1e-4, f"{ratio:.6e} vs {num_ratio:.6e}")
# 6) nested Cholesky: truncating z reproduces the lower-order model on the same draws
chA = of.OAMChannel(j_max=66); chB = of.OAMChannel(j_max=28)
z = np.random.default_rng(3).standard_normal((50, chA.j_list.size))
check("nested truncation equals the J = 28 model", np.allclose(chA.coeffs(z, 1.0, 25), chB.coeffs(z[:, :25], 1.0), rtol=1e-12, atol=1e-14))
# 7) oversampled mode sorter: square default unchanged, rectangular MMSE/ZF identities
chS = of.OAMChannel(j_max=66); chR = of.OAMChannel(j_max=66, rx_modes=tuple(range(-6, 7)))
zz = np.random.default_rng(11).standard_normal((300, chS.j_list.size))
PcS = chS.power_coupling(chS.coeffs(zz, 2.5)); PcR = chR.power_coupling(chR.coeffs(zz, 2.5))
rows = [list(chR.rx_modes).index(m) for m in chS.modes]
check("N_r = 13 coupling contains the square coupling", np.allclose(PcR[:, rows, :], PcS, rtol=1e-12, atol=1e-15))
Ii = chR.scint(np.random.default_rng(12), 300, 2.5); GR = PcR * Ii[:, None, :]
lam, V2 = of.gram_eig(GR)
for snr in (1e3, 1e6, 1e9):
    d1 = of.sinr_mmse(GR, snr); d2 = of.sinr_mmse_eig(lam, V2, snr)
    check(f"eigen MMSE SINR equals direct inverse (N_r = 13, {10*math.log10(snr):.0f} dB)", np.allclose(d1, d2, rtol=1e-8), f"max rel {np.max(np.abs(d1-d2)/d1):.1e}")
    check(f"MMSE SINR >= ZF SNR ({10*math.log10(snr):.0f} dB)", np.all(d2 >= snr * of.zf_gain_eig(lam, V2) * (1 - 1e-9)))
res = []
for b in range(20):
    for m in range(GR.shape[2]):
        A = np.delete(GR[b], m, axis=1); g = GR[b][:, m]
        r = g - A @ np.linalg.lstsq(A, g, rcond=None)[0]
        res.append((r @ r) / of.zf_gain_eig(lam[b], V2[b])[m] - 1)
check("ZF gain equals squared distance from the span of the other columns", np.max(np.abs(res)) < 1e-8, f"max rel {np.max(np.abs(res)):.1e}")
check("gamma_th ~ 8.53 dB", abs(10 * math.log10(of.gamma_th_from_ber()) - 8.53) < 0.01)
print("ALL PASS" if ok else "SOME TESTS FAILED")
