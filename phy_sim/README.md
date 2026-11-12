# Physical-layer simulator (OAM-FSO link, Sections III and VII of the paper)

Monte Carlo simulator of the IM/DD OAM link: correlated Noll-Zernike phase screens (tip/tilt removed,
J = 435, 64 x 128 polar quadrature), aperture-averaged Gamma-Gamma scintillation, thermal and shot noise,
mode sorter with N_r ports, linear MMSE and ZF equalization, 7% hard-decision FEC threshold
(BER 3.8e-3, gamma_th = 8.53 dB). CPU only (NumPy/SciPy); no GPU is used.

## Files

| Script | What it produces |
|---|---|
| `oamfso.py` | Channel model, link budget, receivers (library) |
| `test_oamfso.py` | Unit checks of the channel model |
| `run_modeset.py` | Transmit-mode / sorter-port configurations on common random numbers: outage vs. SNR, diversity slopes, failed-stream counts, failure patterns, pairwise joint failures (anchor loss), effective capacity, 1e6-slot failure traces (`results/modeset_<s2>.json`, `results/traces_<s2>.npz`) |
| `plot_modeset.py` | `results/modeset_summary.json` and the diversity figure |
| `run_phy_results.py` | Stages `analytics`, `jsweep` (Zernike truncation), `sens` (model sensitivity), `power`, `oper` (operating point, crosstalk means), `fig2` |
| `run_alloc.py` | Closed-form power allocation vs. uniform power |
| `run_smallball.py` | Small-ball test of assumption (D2): Pr(delta_m(C) <= eps) and fitted exponents (`results/smallball_<s2>.json`) |
| `run_tolerance.py` | Outage and anchor loss over laser RIN x extra sorter loss (`results/tolerance_<s2>.json`) |
| `run_traces.py`, `run_traces_corr.py` | Temporally correlated slot traces (Gauss-Markov on the Zernike and scintillation variables); `run_traces_corr.py` adds the Kolmogorov temporal correlation rho = exp(-(T_f/tau_c)^(5/3)), the lag-1 correlation of the outage indicator, and packed per-slot outage indicators |
| `launch_all.sh`, `launch_traces.sh` | The exact command lines used for the published results |

## Random number generators and seeds

NumPy `default_rng` (PCG64). `run_modeset.py`: 3301, 3302, 3303 for sigma_R^2 = 0.2, 1.0, 2.5
(+1000 for the point-receiver run). `run_alloc.py`: 4404. `run_phy_results.py`: sens 505, power 404 (other
stages: see the source). `run_smallball.py`: 7701 (sigma_R^2 = 2.5), 7702 (1.0). `run_tolerance.py`: 8801, 8802.
`run_traces_corr.py`: 9100-9119 (tau_c = 2.5 ms), 9200-9209 (tau_c = 10 ms). Multi-process scripts derive
worker seeds with `SeedSequence(seed).spawn(n)`.

## Reproducibility

The results in `results/` were produced twice: on macOS (Apple silicon) on 2026-10-21/22 and on Ubuntu
(Intel Xeon E5-2680 v4, Python 3.14.4, NumPy 2.3.5, SciPy 1.16.3) on 2026-11-12. The failure traces of all
configurations are bit-identical between the two machines. `results/logs/` contains the run logs and
`results/logs/env.txt` the software versions of the second run.

```bash
pip install numpy scipy matplotlib
cd phy_sim
python test_oamfso.py
./launch_all.sh        # full set, about 3 h on 28 cores
```

Single runs, e.g. `python run_modeset.py --s2 2.5 --N 10000000`.
