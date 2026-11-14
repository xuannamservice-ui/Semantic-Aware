# Split-step wave-optics check of the single-screen Zernike channel model

`split_step.py` propagates the eight transmitted OAM modes of TX8 = (-4..-1, 1..4) through a
multi-screen Kolmogorov path and computes the same coupling matrix, received power and MMSE outage
as the single-screen Zernike model of `../oamfso.py` (imported, not modified). CPU only, NumPy/SciPy.

## Method

| Item | Choice |
|---|---|
| Link | lambda = 1550 nm, L = 1500 m, receive aperture D = 0.20 m (from `oamfso`) |
| Sources | LG_{0,l}, l in TX8, waist w0 = sqrt(lambda L / pi) = 2.72 cm (`oamfso.beam_params`), flat phase at z = 0, unit power; plus a Gaussian beacon LG_{0,0} with the same waist |
| Turbulence | constant Cn2 = `oamfso.turbulence(s2)["Cn2"]`; Kolmogorov spectrum 0.023 r0^(-5/3) f^(-11/3), L0 = infinity, l0 = 0 (spectrum truncated by the grid at 1/(2 dx)) |
| Screens | FFT screens with three levels of 3 x 3 subharmonics (Lane et al. 1992; Johansson and Gavel 1994; Schmidt 2010, listing 9.3); real and imaginary parts give two independent screens |
| Layout | Ns = 10 screens at the midpoints of equal 150 m slabs (Ns = 20 as a check). Strengths Cn2_i dz_i = minimum-norm correction of the uniform split that reproduces exactly the plane-wave r0 and the plane-wave Rytov integral 2.25 k^(7/6) sum Cn2_i dz_i (L - z_i)^(5/6) of the continuous path. (oamfso uses the rounded coefficient 1.23 instead of 2.25 x 6/11 = 1.2273, so the screens' Rytov variance is 0.22 % below the nominal value.) |
| Propagation | angular spectrum, exact transfer function exp(i dz (k_z - k)), super-Gaussian absorber exp(-(r / (0.47 N dx))^16) after every partial step; complex64 fields (difference to complex128: < 3e-5 relative in G, < 5e-5 dB in MMSE SINR) |
| Grid | N = 512, dx = 2 mm (1.024 m). Checks: N = 1024, dx = 1 mm (resolution) and N = 1024, dx = 2 mm (extent), sigma_R^2 = 2.5 |
| Receiver | h_rn = sum over the aperture pixels of conj(u_r) U_n exp(-i k r^2 / (2 R(L))) dx^2, u_r = `oamfso.lg_mode(r, ., ., w(L))`, r = -6..6, R(L) = 3000 m (vacuum wavefront curvature for which the sorter is designed); G = abs(h)^2, 13 x 8. No separate Gamma-Gamma factor |
| Tracking variants | `raw`: none. `tt` (primary): receive-side tip/tilt correction by the G-tilt of the beacon over the aperture, the counterpart of removing Zernike j = 2, 3 in oamfso. `ttc`: tt plus recentring of the sorter modes on the beacon centroid (idealized bound; a receive-side mirror cannot remove beam displacement) |
| Detection | `oamfso.gram_eig`, `oamfso.sinr_mmse_eig`, `run_modeset.snr0(M)`, `run_modeset.GTH` (8.53 dB), configurations T8R8, T6R8, T6R11 of `run_modeset.CFG` |
| Zernike model | `oamfso.OAMChannel(modes=TX8, rx_modes=-6..6)`, J = 435, aperture-averaged Gamma-Gamma, `draw()` |

Sampling (Schmidt 2010, ch. 9), with D1 = 16.2 cm (99.99 % of the LG_{0,4} power at z = 0) and
D2 = 0.20 m: dx = 2 mm satisfies dx <= lambda L / (D1' + D2') and N = 512 satisfies
N >= D1'/(2 dx) + D2'/(2 dx) + lambda L/(2 dx^2) for c = 2 and c = 4 at sigma_R^2 = 1.0 and 2.5
(tightest: N_min = 504 for c = 4, sigma_R^2 = 2.5); partial-step limit dz <= dx^2 N / lambda = 1321 m
(largest step 150 m). Per-screen plane-wave Rytov variance 0.037 (sigma_R^2 = 2.5). All numbers are in
`validation.json["sampling"]`.

## Validation (`validation.json`)

* `structure_function`: single screens against 6.88 (r/r0)^(5/3), and against the analytic expectation
  of the generator (discrete spectrum plus subharmonics).
* `zernike_domain`: Noll-Zernike variances per radial order and residual variances Delta_J (J = 1, 3, 10, 21)
  of single screens and of the summed path phase over the 0.20 m aperture, relative to Noll (1976).
* `plane_wave`: point scintillation index of a plane wave (1024 x 1024 grid, central 0.2 m x 0.2 m),
  sigma_R^2 = 0.2, 1.0, 2.5, against Rytov theory and the Andrews-Phillips expression.
* `gaussian_beam_on_axis`: on-axis scintillation of the collimated beacon at sigma_R^2 = 0.2 against weak-
  fluctuation theory (Andrews and Phillips, Theta = Lambda = 0.5).
* `vacuum`: propagated modes against the analytic receive modes without turbulence.

## Seeds (NumPy PCG64)

`SEEDS` in `split_step.py`: split-step main 6101 (sigma_R^2 = 1.0), 6102 (2.5); Ns = 20: 6201, 6202;
grid checks: 6302 (dx = 1 mm), 6312 (2.048 m grid); Zernike model, N = 2e4: 6501, 6502; Zernike model,
N = 1e6: 6601, 6602; validation: 6001 (structure function), SeedSequence(6101) for the Zernike-domain check (it
reuses the screen streams of the first 24 chunks of the sigma_R^2 = 1.0 main run; harmless for a check of screen
statistics, but not an independent sample), 6002 (plane wave), 6003 (Gaussian beam). Split-step runs spawn one SeedSequence child
per chunk of 20 realizations, so the result does not depend on the number of workers.

## Files

| File | Content |
|---|---|
| `split_step.py` | simulator and analysis |
| `run_all.sh` | exact command sequence |
| `validation.json`, `validate.log` | checks above |
| `raw_<tag>_<s2>.npz` | per-realization G (13 x 8) for raw/tt/ttc, aperture power, power left on the grid, beacon tilt and centroid (split-step); G, Gamma-Gamma factors, aperture power (Zernike) |
| `results_<s2>.json` | statistics of every model and variant: mean coupling matrix, self-coupling and crosstalk (dB), received power statistics, MMSE outage at the operating point with Wilson 95 % interval, failed-stream histogram, outage on the 30-100 dB reference-SNR grid |
| `summary_<s2>.txt` | short text summary |

## Rerun

```bash
cd ~/acc24041_rerun/splitstep
./run_all.sh            # about 4.5 h on stt (14 physical cores)
python3 split_step.py time --s2 2.5       # single-realization timing and complex64/complex128 check
```
