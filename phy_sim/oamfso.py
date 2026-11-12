"""OAM-FSO physical-layer model used in the manuscript.

Model (Section III of the paper):
  * M = 8 Laguerre-Gaussian modes (p = 0, m = -4..-1, 1..4), beam waist w0 = sqrt(lambda L / pi),
    observed on a 20 cm receiver aperture at L = 1.5 km.
  * Kolmogorov phase screen on the aperture, expanded in Noll-normalized Zernike polynomials. The
    coefficients are jointly Gaussian with the full Noll covariance (Noll 1976), drawn as
    a = (D_R/r0)^(5/6) L z with L the Cholesky factor of the normalized covariance. Tip/tilt (j = 2, 3)
    is removed by the tracking subsystem, so j = 4..J_MAX.
  * Scintillation: Gamma-Gamma irradiance with aperture-averaged parameters (Andrews and Phillips,
    plane wave) for the collecting diameter of a mode port, D = 2 w(L). The large-scale factor X is
    common to all modes and the small-scale factors Y_n are independent per transmitted mode.
  * IM/DD power-coupling channel G_mn = |h_mn|^2 I_n. Receiver noise: thermal noise of the load and
    post-detection amplifier (noise figure F_n) plus shot noise at the mean received power per mode.
  * IaN and linear-MMSE SINR, outage threshold from the hard-decision FEC limit for OOK.

Two independent implementations of the channel and of the IaN and MMSE SINR were cross-checked on
identical random streams (channel to 2e-15, MMSE SINR to 1e-9 relative at the operating SNR).
"""
import math

import mpmath as mp
import numpy as np
from scipy import integrate, optimize, special
from scipy.stats import norm

# ----------------------------------------------------------------------------- constants
LAMBDA = 1550e-9
K_WAVE = 2.0 * math.pi / LAMBDA
L_LINK = 1500.0
D_R = 0.20
A_R = D_R / 2.0
MODES = (-4, -3, -2, -1, 1, 2, 3, 4)
N_GRID = 64
J_MIN = 4          # tip/tilt (j = 2, 3) compensated by the tracking subsystem
J_MAX = 435        # complete radial orders up to n = 28 (truncation study: stage jsweep)

# link budget (Table "sim_params" of the paper)
P_TOT_DBM = 20.0           # total transmit power
RESPONSIVITY = 0.8         # A/W, InGaAs PIN at 1550 nm
T_K = 300.0                # receiver temperature
R_LOAD = 50.0              # load resistance
F_N_DB = 3.0               # noise figure of the post-detection amplifier
BANDWIDTH = 10e9           # Hz
ATM_LOSS_DB_PER_KM = 0.2   # clear-air attenuation
DEMUX_LOSS_DB = 3.0        # OAM mode-sorter insertion loss
BER_FEC = 3.8e-3           # 7% hard-decision FEC threshold
KB = 1.380649e-23
Q_E = 1.602176634e-19


# ----------------------------------------------------------------------------- link budget
def link_budget(p_tot_dbm=P_TOT_DBM, M=len(MODES)):
    """Per-mode reference electrical SNR of the IM/DD receiver for uniform power allocation.

    gamma0 = (R P_rx)^2 / (sigma_th^2 + sigma_sh^2), sigma_th^2 = 4 k T F_n B / R_L, sigma_sh^2 = 2 q R P_rx B,
    with P_rx = P_tot / M after atmospheric and demux loss (shot noise at the mean received power).
    For OOK with average received power P_rx (levels 0 and 2 P_rx), BER = Q(sqrt(gamma)).
    """
    p_tot = 1e-3 * 10 ** (p_tot_dbm / 10)
    loss_db = ATM_LOSS_DB_PER_KM * L_LINK / 1000 + DEMUX_LOSS_DB
    p_rx = p_tot / M * 10 ** (-loss_db / 10)
    i_p = RESPONSIVITY * p_rx
    s_th = 4 * KB * T_K * 10 ** (F_N_DB / 10) * BANDWIDTH / R_LOAD
    s_sh = 2 * Q_E * i_p * BANDWIDTH
    snr0 = i_p ** 2 / (s_th + s_sh)
    return {"P_tot_W": p_tot, "loss_dB": loss_db, "P_rx_per_mode_W": p_rx, "P_rx_per_mode_dBm": 10 * math.log10(p_rx / 1e-3),
            "photocurrent_A": i_p, "thermal_var_A2": s_th, "shot_var_A2": s_sh, "noise_var_A2": s_th + s_sh,
            "snr0": snr0, "snr0_dB": 10 * math.log10(snr0),
            "snr_thermal_only_dB": 10 * math.log10(i_p ** 2 / s_th), "snr_shot_only_dB": 10 * math.log10(i_p ** 2 / s_sh)}


def gamma_th_from_ber(ber=BER_FEC):
    """OOK: BER = Q(sqrt(gamma)) -> gamma_th = (Q^{-1}(BER))^2."""
    return float(norm.isf(ber) ** 2)


# ----------------------------------------------------------------------------- beam and grid
def beam_params(lam=LAMBDA, L=L_LINK):
    w0 = math.sqrt(lam * L / math.pi)
    zR = math.pi * w0 ** 2 / lam
    w = w0 * math.sqrt(1.0 + (L / zR) ** 2)
    return w0, zR, w


def make_grid(N=N_GRID, a=A_R):
    h = 2.0 * a / N
    c = -a + (np.arange(N) + 0.5) * h
    X, Y = np.meshgrid(c, c, indexing="xy")
    R = np.hypot(X, Y)
    keep = R <= a
    x, y, r = X[keep], Y[keep], R[keep]
    return {"x": x, "y": y, "r": r, "phi": np.arctan2(y, x), "rho": r / a, "dA": h * h, "N": N, "a": a}


def lg_mode(m, r, phi, w):
    am = abs(int(m))
    amp = math.sqrt(2.0 / (math.pi * math.factorial(am))) / w
    return amp * (math.sqrt(2.0) * r / w) ** am * np.exp(-(r / w) ** 2) * np.exp(-1j * m * phi)


# ----------------------------------------------------------------------------- Zernike (Noll)
def noll_nm(j):
    j = int(j)
    n = 0
    while (n + 1) * (n + 2) // 2 < j:
        n += 1
    idx = j - (n * (n + 1) // 2 + 1)
    slots = []
    for m in range(n % 2, n + 1, 2):
        slots += [m] if m == 0 else [m, m]
    m = slots[idx]
    trig = "none" if m == 0 else ("cos" if j % 2 == 0 else "sin")
    return n, m, trig


def zernike_radial(n, m, rho):
    R = np.zeros_like(rho, dtype=float)
    for s in range((n - m) // 2 + 1):
        coef = ((-1) ** s * math.factorial(n - s)
                / (math.factorial(s) * math.factorial((n + m) // 2 - s) * math.factorial((n - m) // 2 - s)))
        R = R + coef * rho ** (n - 2 * s)
    return R


def zernike_noll(j, rho, phi):
    n, m, trig = noll_nm(j)
    R = zernike_radial(n, m, rho)
    if m == 0:
        return math.sqrt(n + 1) * R
    f = np.cos(m * phi) if trig == "cos" else np.sin(m * phi)
    return math.sqrt(n + 1) * R * math.sqrt(2.0) * f


def noll_cn(n):
    g = special.gamma
    return float(0.0072 * (n + 1) * math.pi ** (8 / 3) * g(14 / 3) * g(n - 5 / 6) / (g(17 / 6) ** 2 * g(n + 23 / 6)))


def noll_residual(J):
    """Residual phase variance Delta_J / (D/r0)^(5/3) after removing piston and the Zernike terms j <= J
    (Noll 1976): Delta_J = sum_{j > J} c_j, with sum_{n >= 1} (n + 1) c_n evaluated to n = 2e5."""
    n = np.arange(1, 200001, dtype=float)
    lc = (math.log(0.0072 * math.pi ** (8 / 3)) + special.gammaln(14 / 3) + np.log(n + 1) + special.gammaln(n - 5 / 6)
          - 2 * special.gammaln(17 / 6) - special.gammaln(n + 23 / 6))
    total = float(np.sum((n + 1) * np.exp(lc)))
    return total - sum(noll_cn(noll_nm(j)[0]) for j in range(2, J + 1))


def noll_cov(j_min=J_MIN, j_max=J_MAX):
    """Covariance E[a_j a_j'] / (D/r0)^(5/3) of the Noll-normalized Zernike coefficients of a Kolmogorov
    phase screen (Noll 1976), j, j' = j_min..j_max. Nonzero only for equal azimuthal order m and, for
    m != 0, equal cos/sin parity; the diagonal equals noll_cn(n). Signed Gamma functions are used, since
    Gamma((n - n' + 17/3)/2) is negative for some |n - n'| >= 6."""
    g = special.gamma
    K = 0.0072 * math.pi ** (8 / 3) * g(14 / 3)
    js = np.arange(j_min, j_max + 1)
    nm = [noll_nm(j) for j in js]
    C = np.zeros((js.size, js.size))
    for i, (n, m, t) in enumerate(nm):
        for k, (n2, m2, t2) in enumerate(nm):
            if m != m2 or (m != 0 and t != t2):
                continue
            num = g((n + n2 - 5 / 3) / 2)
            den = g((n - n2 + 17 / 3) / 2) * g((n2 - n + 17 / 3) / 2) * g((n + n2 + 23 / 3) / 2)
            C[i, k] = K * (-1) ** ((n + n2 - 2 * m) // 2) * math.sqrt((n + 1) * (n2 + 1)) * num / den
    return C


# ----------------------------------------------------------------------------- turbulence
def turbulence(sigmaR2, lam=LAMBDA, L=L_LINK, D=D_R):
    k = 2 * math.pi / lam
    Cn2 = sigmaR2 / (1.23 * k ** (7 / 6) * L ** (11 / 6))
    r0 = (0.423 * k ** 2 * Cn2 * L) ** (-3 / 5)
    return {"Cn2": Cn2, "r0": r0, "D_over_r0": D / r0}


def gg_point(sigmaR2):
    """Point-receiver plane-wave Gamma-Gamma parameters (Andrews and Phillips)."""
    s = float(sigmaR2); s65 = s ** 1.2
    a = 1 / math.expm1(0.49 * s / (1 + 1.11 * s65) ** (7 / 6))
    b = 1 / math.expm1(0.51 * s / (1 + 0.69 * s65) ** (5 / 6))
    return a, b


def gg_aperture(sigmaR2, D=None, lam=LAMBDA, L=L_LINK):
    """Aperture-averaged plane-wave Gamma-Gamma parameters (Andrews and Phillips), d^2 = k D^2/(4L).

    D defaults to the collecting diameter of a mode port, 2 w(L).
    """
    if D is None:
        D = 2 * beam_params(lam, L)[2]
    k = 2 * math.pi / lam
    d2 = k * D ** 2 / (4 * L)
    s = float(sigmaR2); s65 = s ** 1.2
    lx = 0.49 * s / (1 + 0.65 * d2 + 1.11 * s65) ** (7 / 6)
    ly = 0.51 * s * (1 + 0.69 * s65) ** (-5 / 6) / (1 + 0.90 * d2 + 0.62 * d2 * s65)
    return 1 / math.expm1(lx), 1 / math.expm1(ly)


def scint_index(a, b):
    return 1 / a + 1 / b + 1 / (a * b)


# ----------------------------------------------------------------------------- channel
class OAMChannel:
    def __init__(self, N=N_GRID, j_min=J_MIN, j_max=J_MAX, modes=MODES, a=A_R, D_scint=None, point_receiver=False,
                 correlated=True, shared_pm=False, rx_modes=None):
        """modes: transmitted LG_{0,l} modes (one data stream each). rx_modes: LG_{0,l} projections of the
        mode sorter (one photodiode each); defaults to the transmitted set (square N_r = M receiver)."""
        self.w0, self.zR, self.w = beam_params()
        self.grid = make_grid(N, a)
        self.modes = np.asarray(modes, dtype=int)
        self.M = self.modes.size
        self.rx_modes = self.modes.copy() if rx_modes is None else np.asarray(rx_modes, dtype=int)
        self.Nr = self.rx_modes.size
        U = np.stack([lg_mode(m, self.grid["r"], self.grid["phi"], self.w) for m in self.modes])
        Ur = np.stack([lg_mode(m, self.grid["r"], self.grid["phi"], self.w) for m in self.rx_modes])
        Qm = np.conj(Ur)[:, None, :] * U[None, :, :] * self.grid["dA"]
        self.QT = np.ascontiguousarray(Qm.reshape(self.Nr * self.M, -1).T)
        self.QTr = np.ascontiguousarray(self.QT.real)
        self.QTi = np.ascontiguousarray(self.QT.imag)
        self.j_list = np.arange(j_min, j_max + 1)
        self.noll = [noll_nm(j) for j in self.j_list]
        self.Z = np.stack([zernike_noll(j, self.grid["rho"], self.grid["phi"]) for j in self.j_list])
        self.c_of_j = np.array([noll_cn(n) for (n, _, _) in self.noll])
        self.correlated = correlated
        # a = (D/r0)^(5/6) z Lc^T; Lc is lower triangular, so the first k coefficients depend only on the
        # first k normals and truncating z and Lc gives the lower-order model on the same draws.
        self.Lc = np.linalg.cholesky(noll_cov(j_min, j_max)) if correlated else np.diag(np.sqrt(self.c_of_j))
        self.D_scint = 2 * self.w if D_scint is None else D_scint
        self.point_receiver = point_receiver
        self.shared_pm = shared_pm          # sensitivity case: Y_{+m} = Y_{-m}

    def gg_params(self, sigmaR2):
        return gg_point(sigmaR2) if self.point_receiver else gg_aperture(sigmaR2, self.D_scint)

    def coeffs(self, z, sigmaR2, nj=None):
        """Zernike coefficients j = j_min..j_min+nj-1 from standard normals z (leading nj columns)."""
        nj = self.j_list.size if nj is None else nj
        return (z[:, :nj] @ self.Lc[:nj, :nj].T) * turbulence(sigmaR2)["D_over_r0"] ** (5 / 6)

    def power_coupling(self, a):
        """|h_mn|^2 = |sum_k conj(u_m) u_n exp(j Phi) dA|^2 (receive port m, transmitted mode n), evaluated with
        real arithmetic; shape (B, N_r, M)."""
        Phi = a @ self.Z[:a.shape[1]]
        c, s = np.cos(Phi), np.sin(Phi)
        hr = c @ self.QTr - s @ self.QTi
        hi = c @ self.QTi + s @ self.QTr
        return (hr ** 2 + hi ** 2).reshape(a.shape[0], self.Nr, self.M)

    def scint(self, rng, B, sigmaR2):
        al, be = self.gg_params(sigmaR2)
        X = rng.gamma(al, 1 / al, size=B)
        Y = rng.gamma(be, 1 / be, size=(B, self.M))
        if self.shared_pm:
            idx = {int(m): i for i, m in enumerate(self.modes)}
            for i, m in enumerate(self.modes):
                if m < 0:
                    Y[:, i] = Y[:, idx[-int(m)]]
        return X[:, None] * Y

    def draw(self, rng, B, sigmaR2, return_parts=False):
        z = rng.standard_normal((B, self.j_list.size))
        Pc = self.power_coupling(self.coeffs(z, sigmaR2))
        I = self.scint(rng, B, sigmaR2)
        G = Pc * I[:, None, :]
        return (G, Pc, I) if return_parts else G


# ----------------------------------------------------------------------------- receivers
def _pow(P, M):
    P = np.asarray(P, dtype=float)
    return np.full(M, float(P)) if P.ndim == 0 else P


def sinr_ian(G, P, sigma2=1.0):
    M = G.shape[-1]; P = _pow(P, M)
    G2 = G * G
    sig = np.diagonal(G2, axis1=-2, axis2=-1) * P
    interf = np.einsum("...mn,...n->...m", G2 * (1 - np.eye(M)), P)
    return sig / (interf + sigma2)


def sinr_mmse(G, P, sigma2=1.0):
    """gamma_m = P_m / (sigma2 [A^{-1}]_mm) - 1 with A = G^T G + sigma2 diag(1/P) (equal to the
    explicit MMSE-SINR expression of the paper; verified to 1e-12). G may be N_r x M with N_r >= M."""
    M = G.shape[-1]; P = _pow(P, M)
    GT = np.swapaxes(G, -1, -2)
    A = GT @ G + sigma2 * np.diag(1.0 / P)
    d = np.diagonal(np.linalg.inv(A), axis1=-2, axis2=-1)
    return P / (sigma2 * d) - 1.0


def gram_eig(G):
    """Eigen-decomposition of the Gram matrix G^T G of an (..., N_r, M) coupling matrix: returns
    (lam, V2) with lam (..., M) >= 0 and V2[..., m, k] = V[..., m, k]^2. Used to evaluate the MMSE and ZF
    SINR at many SNR values from one factorization."""
    lam, V = np.linalg.eigh(np.swapaxes(G, -1, -2) @ G)
    return np.maximum(lam, 0.0), V * V


def sinr_mmse_eig(lam, V2, snr):
    """Uniform-power MMSE SINR from gram_eig: gamma_m = 1 / sum_k V_mk^2 / (1 + snr lam_k) - 1."""
    return 1.0 / np.einsum("...mk,...k->...m", V2, 1.0 / (1.0 + snr * lam)) - 1.0


def zf_gain_eig(lam, V2):
    """Zero-forcing SNR per unit transmit SNR, 1 / [(G^T G)^{-1}]_mm: the squared distance of column m of G
    from the span of the other columns."""
    return 1.0 / np.einsum("...mk,...k->...m", V2, 1.0 / np.maximum(lam, 1e-300))


# ----------------------------------------------------------------------------- equivalent Gamma-Gamma surrogate
def fit_equivalent_gg(gamma, ratio):
    """Lemma 1 of the paper: moment matching of the equivalent irradiance I = sqrt(gamma / gbar'),
    gbar' = (E sqrt(gamma))^2, with alpha'/beta' = ratio. Returns (alpha', beta', gbar')."""
    g = np.asarray(gamma, dtype=float)
    m1 = np.mean(np.sqrt(g)); m2 = np.mean(g)
    gbar = m1 ** 2
    si = m2 / gbar - 1.0
    r = ratio
    a = (1 + r + math.sqrt((1 + r) ** 2 + 4 * si * r)) / (2 * si)
    return a, a / r, gbar


def gg_cdf_I(x, a, b):
    """CDF of the unit-mean Gamma-Gamma irradiance: G^{2,1}_{1,3}(ab x | 1; a, b, 0)/(Gamma(a)Gamma(b))."""
    return float(mp.meijerg([[1], []], [[a, b], [0]], a * b * x) / (mp.gamma(a) * mp.gamma(b)))


def gg_quantile_I(p, a, b):
    """Inverse CDF of the unit-mean Gamma-Gamma irradiance (bisection in log x)."""
    lo, hi = 1e-12, 50.0
    for _ in range(120):
        mid = math.sqrt(lo * hi)
        if gg_cdf_grid([mid], a, b)[0] > p:
            hi = mid
        else:
            lo = mid
    return math.sqrt(lo * hi)


def surrogate_outage(gth, a, b, gbar):
    """Per-mode outage of the surrogate: P(gamma < gth) = F_I(sqrt(gth/gbar))."""
    return gg_cdf_I(math.sqrt(gth / gbar), a, b)



def gg_pdf_I(x, a, b):
    """Unit-mean Gamma-Gamma irradiance pdf, evaluated in the log domain (scaled Bessel K, mpmath fallback)."""
    if x <= 0:
        return 0.0
    z = 2 * math.sqrt(a * b * x)
    lg = (math.log(2) + 0.5 * (a + b) * math.log(a * b) + (0.5 * (a + b) - 1) * math.log(x)
          - special.gammaln(a) - special.gammaln(b))
    nu = abs(a - b)
    kv = special.kve(nu, z)
    if np.isfinite(kv) and kv > 0:
        val = lg - z + math.log(kv)
    elif nu >= 1:
        val = lg + float(log_besselk_small(nu, z))                           # small-argument expansion
    else:
        val = lg + float(mp.log(mp.besselk(nu, z)))
    return math.exp(val) if val < 700 else float("inf")


def gg_cdf_fast(x, a, b):
    """CDF of the unit-mean Gamma-Gamma irradiance at a scalar x (adaptive Simpson grid, gg_cdf_grid;
    agrees with the Meijer-G form gg_cdf_I to about 1e-9 relative)."""
    if x <= 0:
        return 0.0
    return float(gg_cdf_grid([x], a, b)[0])


def log_besselk_small(nu, z, terms=8):
    """log K_nu(z) for nu >= 1 and z << nu, where scipy's scaled Bessel function overflows:
    K_nu(z) = (Gamma(nu)/2)(2/z)^nu sum_k (z^2/4)^k / (k! (1-nu)_k) + O((z/2)^nu)."""
    z = np.asarray(z, dtype=float)
    base = special.gammaln(nu) - math.log(2) + nu * np.log(2 / z)
    if nu < 20:
        return base
    t = np.ones_like(z); S = np.ones_like(z); q = z * z / 4
    for k in range(1, terms + 1):
        t = t * q / (k * (k - nu))
        S = S + t
    return base + np.log(S)


def gg_logpdf_vec(x, a, b):
    """Log of the unit-mean Gamma-Gamma irradiance pdf for an array x > 0. Where the scaled Bessel
    function overflows (large order nu = |a - b| and z << nu), the leading small-argument term
    log K_nu(z) = log Gamma(nu) - log 2 + nu log(2/z) is used; mpmath is the fallback otherwise."""
    x = np.asarray(x, dtype=float)
    z = 2 * np.sqrt(a * b * x)
    nu = abs(a - b)
    lg = (math.log(2) + 0.5 * (a + b) * math.log(a * b) + (0.5 * (a + b) - 1) * np.log(x)
          - special.gammaln(a) - special.gammaln(b))
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        kv = special.kve(nu, z)
        out = lg - z + np.log(kv)
    bad = ~np.isfinite(out)
    if bad.any():
        if nu >= 1:
            out[bad] = lg[bad] + log_besselk_small(nu, z[bad])
        else:
            for i in np.flatnonzero(bad):
                out.flat[i] = lg.flat[i] + float(mp.log(mp.besselk(nu, z.flat[i])))
    return out


def gg_cdf_grid(xs, a, b, h_rate=0.02):
    """CDF of the unit-mean Gamma-Gamma irradiance at the points xs, by composite Simpson integration
    of x f(x) over u = ln x. The lower limit is placed where the integrand is negligible (the lower
    tail decays as x^min(a,b), and the body of ln I has standard deviation sd), and the step keeps
    h * max(min(a,b), 1/sd) <= h_rate."""
    xs = np.atleast_1d(np.asarray(xs, dtype=float))
    out = np.zeros_like(xs)
    lam = min(a, b)
    mu = special.digamma(a) - math.log(a) + special.digamma(b) - math.log(b)
    sd = math.sqrt(special.polygamma(1, a) + special.polygamma(1, b))
    rate = max(lam, 1.0 / sd)
    for i, x in enumerate(xs):
        if x <= 0:
            continue
        ux = math.log(x)
        lo = min(ux - 32.0 / lam, mu - 12.0 * sd)
        n = int(min(40001, max(401, (ux - lo) * rate / h_rate)))
        n += 1 - n % 2
        u = np.linspace(lo, ux, n)
        g = np.exp(gg_logpdf_vec(np.exp(u), a, b) + u)
        h = u[1] - u[0]
        out[i] = h / 3 * (g[0] + g[-1] + 4 * g[1:-1:2].sum() + 2 * g[2:-1:2].sum())
    return np.minimum(out, 1.0)


TAIL_PROBS = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)


def fit_tail_gg(gamma, probs=TAIL_PROBS, min_count=100, init=None):
    """Tail-matched equivalent Gamma-Gamma model: (alpha', beta', gbar') minimizing
    sum_p [log10 F_sur(q_emp(p)) - log10 p]^2 over lower-tail probabilities p that have at least
    min_count samples below the empirical quantile q_emp(p)."""
    g = np.asarray(gamma, dtype=float)
    n = g.size
    ps = [p for p in probs if p * n >= min_count]
    q = np.quantile(g, ps)
    if init is None:
        a0, b0, gb0 = fit_equivalent_gg(g, 1.0)
        init = (a0, b0, gb0)

    lp = np.log10(ps)

    def obj(t):
        a, b, gb = math.exp(t[0]), math.exp(t[1]), math.exp(t[2])
        if a > 400 or b > 400 or a < 0.05 or b < 0.05:
            return 1e6
        F = gg_cdf_grid(np.sqrt(q / gb), a, b)
        return float(np.sum((np.log10(np.maximum(F, 1e-300)) - lp) ** 2))

    best = None
    for scale in (1.0, 0.5, 0.25):
        t0 = np.log([max(init[0] * scale, 0.1), max(init[1] * scale, 0.1), init[2]])
        r = optimize.minimize(obj, t0, method="Nelder-Mead", options={"xatol": 1e-6, "fatol": 1e-10, "maxiter": 4000})
        if best is None or r.fun < best.fun:
            best = r
    a, b, gb = np.exp(best.x)
    return float(a), float(b), float(gb), float(best.fun), list(ps)

# ----------------------------------------------------------------------------- effective capacity
def ec_norm_mc(c, s):
    """Monte Carlo normalized effective capacity -(1/s) ln E[exp(-s c)], c in bit/s/Hz (log-sum-exp)."""
    c = np.asarray(c, dtype=float)
    x = -s * c
    mx = x.max()
    return -(mx + math.log(np.mean(np.exp(x - mx)))) / s
