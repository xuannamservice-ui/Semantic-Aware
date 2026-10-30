"""Bivariate Gauss-Markov surrogate of the anchor-pair losses, recalibrated to the Monte Carlo
marginal of Table loss_patterns (strong turbulence, square sorter): P(+1 lost) = 0.018,
P(+1 and -2 lost) = 1.75e-3. The marginal of mode -2 is not tabulated; it is set equal to that of
+1 (conservative, since modes +-2 fail less often than +-1)."""
import math, numpy as np
p1 = p2 = 0.018; pj = 1.75e-3; tau_c, Tf = 2.5, 1.0; N = 10_000_000; seed = 7
Phi = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))
def Phi_inv(p, lo=-10, hi=10):
    for _ in range(200):
        m = (lo + hi) / 2; lo, hi = (m, hi) if Phi(m) < p else (lo, m)
    return (lo + hi) / 2
t1, t2 = Phi_inv(p1), Phi_inv(p2)
x = np.linspace(-12, t1, 40001); phi = np.exp(-x**2 / 2) / math.sqrt(2 * math.pi)
def joint(r):
    s = math.sqrt(1 - r * r); v = np.array([Phi((t2 - r * xi) / s) for xi in x[::40]])
    return np.trapz(phi[::40] * v, x[::40])
lo, hi = 0.0, 0.99
for _ in range(60):
    m = (lo + hi) / 2; lo, hi = (m, hi) if joint(m) < pj else (lo, m)
r = (lo + hi) / 2
rho = math.exp(-(Tf / tau_c) ** (5 / 3))
rng = np.random.default_rng(seed)
e1 = rng.standard_normal(N); e2 = r * e1 + math.sqrt(1 - r * r) * rng.standard_normal(N)
a = math.sqrt(1 - rho * rho)
z1 = np.empty(N); z2 = np.empty(N); c1 = rng.standard_normal(); c2 = r * c1 + math.sqrt(1 - r * r) * rng.standard_normal()
for i in range(N):
    c1 = rho * c1 + a * e1[i]; c2 = rho * c2 + a * e2[i]; z1[i] = c1; z2[i] = c2
L1 = z1 < t1; LP = L1 & (z2 < t2)
def bursts(s):
    d = np.diff(np.concatenate(([0], s.astype(np.int8), [0]))); return np.where(d == -1)[0] - np.where(d == 1)[0]
b1, bp = bursts(L1), bursts(LP)
print(f"r_cross={r:.4f} rho={rho:.4f} N={N:.0e} seed={seed}")
print(f"single-anchor loss {L1.mean():.5f}  pair loss {LP.mean():.3e}  gain {L1.mean()/LP.mean():.1f}")
print(f"pair bursts: n={len(bp)} mean={bp.mean():.2f} >=4: {np.sum(bp>=4)} ({np.mean(bp>=4):.4f}) max={bp.max()}")
print(f"single bursts: mean={b1.mean():.2f} max={b1.max()}")
print(f"queue growth rate pair - delta = {LP.mean()-1e-4:.3e} per slot")
