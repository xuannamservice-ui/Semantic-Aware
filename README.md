# Semantic-Aware Generative Error Concealment for Resilient Real-Time 3D Gaussian Splatting Streaming over Turbulent FSO Backhaul

Code, simulation data and measured results for the paper of the same title
(P. V. M. Tu, N. X. Nam, V. V. Thuong, T. V. Huu, Creative Technologies and Simulation Lab, Posts and
Telecommunications Institute of Technology, Hanoi).

The repository has two independent parts that meet in one interface, the **per-slot failure pattern** of the
optical link (which of the parallel streams failed in a slot):

| Part | Folder | What it does | Hardware |
|---|---|---|---|
| Optical link | [`phy_sim/`](phy_sim/) | Wave-optics Monte Carlo of an intensity-modulated OAM free-space link with a mode sorter and linear MMSE equalization; writes outage statistics and **failure-pattern traces** | CPU only |
| Split-step check | [`phy_sim/splitstep/`](phy_sim/splitstep/) | Multi-screen wave-optics propagation of the same link (10 phase screens over 1.5 km) as an independent check of the single-screen model; also writes failure patterns | CPU only |
| Codec | [`codec/`](codec/) | Concealment-aware layered 3D Gaussian Splatting codec and generative concealment network; trains on Mip-NeRF 360, Tanks and Temples and Deep Blending scenes and evaluates the rendering quality under the failure traces | NVIDIA GPU |
| HAC baseline | [`hac/`](hac/) | Scripts that run the official HAC code on the same scenes and apply the same failure patterns to its anchors | NVIDIA GPU |

The codec does not depend on the optical model: it only reads failure-pattern traces (`phy_sim/results/traces_*.npz`,
`phy_sim/results/correlated/`). Traces from any other multi-stream link can be used in their place.

---

## 1. Repository layout

```
phy_sim/                      optical-link simulator (Python, NumPy/SciPy)
  oamfso.py                   channel model, link budget, receivers (library)
  run_modeset.py              mode/port configurations on common random numbers -> outage, patterns, traces
  run_phy_results.py          operating point, effective capacity reference, truncation, sensitivity, power
  run_alloc.py                closed-form power allocation versus uniform power
  run_smallball.py            test of the small-ball assumption (D2) of the diversity analysis
  run_tolerance.py            outage versus laser RIN and extra sorter insertion loss
  run_jconv.py                convergence in the number of Zernike terms J and in the aperture grid
  run_traces.py, run_traces_corr.py   temporally correlated slot traces (Model B), with per-mode patterns
  wdm_reference.py            WDM reference: M wavelengths on one Gaussian beam, no crosstalk
  launch_*.sh                 exact command lines used for the published results
  results/                    all outputs (JSON, NPZ) and run logs; see section 4
  splitstep/                  split-step multi-screen simulation (split_step.py, run_all.sh), validation of the screens,
                              outage results, failure patterns of the realizations (ss_patterns.py -> traces_ss2.5.npz)
codec/                        layered 3DGS codec and concealment (PyTorch, gsplat)
  scene_data.py               COLMAP reader, train/test split (every 8th image is a test view)
  train_3dgs.py               reference 3DGS model per scene (MCMC densification, 500,000 Gaussians)
  anchor_model.py             anchor-based (Scaffold-GS-style) representation, quantization
  train_layered.py            final codec: layered representation + entropy model + concealment network (joint)
  conceal_net.py              concealment networks (LiteConcealNet, EdgeConcealNet, reference transformer)
  streaming.py                tokens, token-to-stream mapping policies, loss application
  train_lite.py               concealment network on a frozen representation (deployment variants)
  eval_layered.py             outage-slot evaluation on the failure traces (cold-start view)
  eval_trajectory.py          camera-path evaluation with an edge cache
  eval_frames.py              frame-level evaluation: concealment vs. erasure coding and in-frame retransmission
  bench_latency.py            latency of gathering, concealment, decoding and rendering (CUDA events)
  new_anchors.py              anchors that become visible for the first time per frame along the capture path
  queue_*.sh, run_*.sh        job queues used on the shared GPU server (memory guards included)
  results/                    per-scene metrics and evaluations (JSON); see section 4
  archive/                    first-generation (non-layered) codec scripts kept for reference; not used for the paper
hac/                          HAC baseline: run_one.sh (training, encoding, decoding with the official code and settings),
                              hac_trace_eval.py (HAC under the failure traces, drop and IDW), run_trace_eval*.sh
legacy/                       earlier surrogate script and empty measurement templates; superseded, not used
```

## 2. Installation

Physical layer (CPU):
```bash
pip install numpy scipy matplotlib
```
Codec (GPU, tested with PyTorch 2.10/CUDA 12.8 on NVIDIA L40S and PyTorch 2.14/CUDA 12.6 on GTX 1070):
```bash
pip install -r requirements.txt     # torch, gsplat, lpips, imageio, plyfile, ...
```
`gsplat` compiles its CUDA kernels on first use (needs `nvcc`). On GPUs older than Volta (sm_61) the kernels use
`cooperative_groups::labeled_partition`, which is not available; replacing `cg::labeled_partition(warp, gid)` by
`cg::tiled_partition<1>(warp)` in `gsplat/cuda/csrc/Projection*.cu` gives per-thread atomics with identical results
(checked with finite differences); this was used only for the GTX 1070 runs.

Datasets: Mip-NeRF 360 (`codec/fetch_mipnerf360.py` downloads only `images_4`/`images_2` and `sparse/0` of the needed
scenes by HTTP range requests); Tanks and Temples and Deep Blending as distributed with the 3DGS code
(`tandt_db.zip`). Expected layout: `codec/data/360_v2/<scene>`, `codec/data/tandt/<scene>`, `codec/data/db/<scene>`.

## 3. Reproducing the results

### 3.1 Optical link (`phy_sim/`)
```bash
cd phy_sim
python test_oamfso.py                                         # unit checks of the channel model
python run_modeset.py --s2 2.5 --N 10000000                   # 1e7 slots, sigma_R^2 = 2.5 (seed 3303)
python run_modeset.py --s2 2.5 --N 10000000 --jmax 780 --suffix _J780
python run_smallball.py --s2 2.5 --N 20000000 --workers 12    # assumption (D2)
python run_tolerance.py --s2 2.5 --N 10000000                 # RIN x extra loss
python run_jconv.py --s2 2.5 --N 10000000 --workers 24        # J and grid convergence
python run_traces_corr.py --s2 2.5 --tau 10 --chains 2000 --slots 250 --seed 9700 --rho-model kolmogorov --tag _pat0
python wdm_reference.py
python run_beat.py 2.5 200000                                 # common carrier and wavelength reuse (W = 4, 2, 1), seed 606
for i in 1 2 3 4 5; do python run_reuse.py chunk 0.2 $((3400+i)) 2000000 reuse_chunks/c$i.npz; done
python run_reuse.py merge 0.2 0.2w reuse_chunks/c*.npz      # 8 and 6 modes on 4 carriers, 1e7 slots, traces_0.2w.npz
./launch_all.sh                                               # the full set as published
cd splitstep && ./run_all.sh                                  # split-step check (several hours; run times in run_all.log)
python ss_patterns.py --s2 2.5 --variant tt                   # failure patterns of the split-step realizations
```
All generators are NumPy `default_rng` (PCG64) with fixed seeds (listed in [`phy_sim/README.md`](phy_sim/README.md)).
The physical-layer results were computed twice, on macOS (Apple silicon) and on Ubuntu (Intel Xeon E5-2680 v4,
Python 3.14.4, NumPy 2.3.5, SciPy 1.16.3); all failure traces and event counts are bit-identical (SHA-256 per array in
`phy_sim/results/reproducibility.json`), floating-point summaries agree to about 1e-11 (the weak-turbulence Gamma-Gamma tail fit, an ill-conditioned optimisation, is the
only exception).

### 3.2 Codec (`codec/`)
```bash
cd codec
python train_3dgs.py --scene garden                                   # reference 3DGS (about 5 min on an idle L40S)
python train_layered.py --scene garden --tag _v2a_r1 --anchors 250000 --feat-dim 48 --base-dim 12 \
       --net lite --d 96 --layers 2 --aware 1 --rate-lambda 5e-5      # final joint design
python train_layered.py --scene garden --tag _v2s_r1 ... --aware 0     # separate design (ablation)
python eval_layered.py --scene garden --tag _v2a_r1 --cfg T8R8 --s2 2.5 --n-pat 6
python eval_frames.py  --scene garden --tag _v2a_r1 --cfg T8R8 --tau 10 --n-anchor 4 --chains 10
python eval_frames.py  --scene garden --tag _v2a_r1 --cfg T8R8 --tau 10 --n-anchor 4 --chains 10 --stride 4   # faster camera
python eval_layered.py --scene garden --tag _v2a_r1 --cfg T8R8 --s2 ss2.5 --phy ../phy_sim/splitstep           # split-step patterns
python bench_latency.py --out results/bench.json --lost 1000,5000,10000,20000,30000
python eval_layered.py --scene garden --tag _v2a_r1 --cfg T8R11W4 --s2 0.2w --phy ../phy_sim/results   # wavelength-reuse link
python eval_frames.py  --scene garden --tag _v2a_r1 --cfg T8R8 --tau 10 --n-anchor 4 --chains 50 --eps 10    # ARQ without budget, full replication
./queue_exp.sh jobs_basedim.txt                                       # base layer with 6 and 24 feature channels
```
`queue_exp.sh jobs_all_v2.txt` runs the trainings and evaluations of all scenes with memory guards for a shared server.

## 4. Data files

### Optical link (`phy_sim/results/`)
| File | Content |
|---|---|
| `modeset_<s2>.json` | per configuration (`T<M>R<N_r>`: M transmitted modes, N_r sorter ports): MMSE/ZF outage counts on an SNR grid and at the link-budget operating point (`op`), failed-stream histogram, failure-pattern counts, pairwise joint failures (anchor loss), per-mode failures, sum spectral efficiency |
| `modeset_<s2>_J780.json` | the same with 780 Zernike terms |
| `traces_<s2>.npz` | failure pattern of the first 1e6 slots of every configuration (uint8; bit i set if the i-th transmitted mode failed) |
| `modeset_summary.json` | outage, slopes (diversity), pattern statistics per configuration and turbulence level |
| `smallball_<s2>*.json` | Pr(delta_m(C) <= eps) and fitted local exponents per window |
| `tolerance_<s2>.json` | outage and best anchor pair versus RIN and extra sorter loss |
| `jconv_<s2>.json` | outage versus J (435, 595, 780) and grid (64, 96), paired differences |
| `wdm_reference.json` | WDM reference outage (all channels fail together) |
| `beat_<s2>.json` | outage with separate carriers, 4 or 2 shared carriers and one common carrier (intermodal beat terms), 2e5 slots |
| `modeset_0.2w.json`, `traces_0.2w.npz` | wavelength reuse at Rytov variance 0.2 (configurations `T8R11W4`, `T6R11W4`): outage, pattern histogram, best anchor sets, trace of 1e6 slots |
| `correlated/traces_corr_*` | Model B: temporally correlated slots (tau_c = 2.5, 10 ms); `*_pat*.npz` store per-slot failure patterns |
| `oper_*`, `analytics`, `alloc_*`, `sens_*`, `jsweep_*`, `power_*` | operating point, effective-capacity reference, power allocation, sensitivity, truncation |
| `reproducibility.json` | SHA-256 of every trace array from the two machines and the identity check |
| `logs/` | run logs; `env.txt` software versions |

### Codec (`codec/results/<scene>/`)
| File | Content |
|---|---|
| `metrics_3dgs.json` | reference 3DGS: test PSNR/SSIM/LPIPS, number of Gaussians, training log |
| `metrics_layered_<tag>.json` | layered codec: clean test quality, bits per anchor (empirical entropy and entropy-model code length), scene size by layer |
| `eval_layered_<tag>_<cfg>_<s2>.json` | outage-slot evaluation: per (view, failure pattern, mapping policy, method) PSNR/SSIM/LPIPS and PSNR against the loss-free rendering; `summary` per policy and method |
| `traj_*.json`, `frames_*.json` | camera-path and frame-level evaluations (edge cache; concealment, retransmission, erasure coding) |
| `../bench_l40s_idle.json` | latency on an idle L40S (2,000 timed runs per stage) |

Tags: `_v2a_r<k>` final joint design (`r1`: rate weight 5e-5, `r0`: 0, `r2`: 2e-4); `_v2s_r1` separate design.
Tags `_a150`, `_s150`, `_a80`, `_s80` and `metrics_anchor.json` come from development iterations of the codec
(150k/80k anchors, reference transformer) and are kept for completeness.

## 5. Main results available in this repository

Optical link, Rytov variance 2.5, MMSE outage at the operating point (10^7 slots):

| Configuration | J = 435 | J = 780 | Note |
|---|---|---|---|
| 8 modes / 8 ports | 1.456e-1 | 1.496e-1 | |
| 6 modes / 8 ports | 2.683e-3 | 2.915e-3 | two base-layer copies lost together: 2.4e-5 |
| 8 modes / 11 ports | 2.337e-3 | 2.573e-3 | |
| 6 modes / 11 ports | 9.17e-5 | 1.083e-4 | |
| WDM, 6 wavelengths, one Gaussian beam | | 3.3e-17 (Gamma-Gamma CDF) | all channels fail together |

Split-step check (10 phase screens, 2e4 realizations, tip-tilt corrected, `phy_sim/splitstep/`): T6R8 3.3e-3 and T6R11 5e-5
agree with the single-screen model within the 95% intervals; for T8R8 the single-screen model is optimistic (0.228 against 0.146).

Codec, eleven scenes (Mip-NeRF 360, Tanks and Temples, Deep Blending), final joint design, outage slots of the 8-mode/8-port
trace, interleaved mapping (`codec/results/<scene>/eval_layered_v2a_r1_T8R8_2.5.json`):

| | Loss-free | Drop | IDW | Concealment network |
|---|---|---|---|---|
| Mean PSNR over 11 scenes (dB) | 27.86 | 22.49 | 24.50 | 27.34 |

The network is 2.99 dB above IDW (95% interval [2.81, 3.17] from a bootstrap over the 338 test views) and better in 99.8% of the
pairs of test view and failure pattern; joint training adds 0.96 to 1.85 dB over a separately trained network.

Frame level with an edge cache (11 scenes, 3,380 test frames, T8R8, tau_c = 10 ms, `frames_*.json`): frames below 30 dB against
the loss-free rendering: none 22, concealment 0, retransmission 8, (8,7) erasure code 10 (redundancy at most 1/6 and 1/7). Without a redundancy budget
(`*_eps10.json`), retransmission and full replication of the enhancement layer on all eight streams leave no frame below 30 or 35 dB,
whereas concealment alone leaves 9 below 35 dB; replication fits into one slot only up to about 15,700 new anchors per frame.

Common carrier and wavelength reuse (`phy_sim/results/beat_*.json`, `modeset_0.2w.json`): with one carrier for all modes the intermodal
beat terms drive the outage above 0.97 for every configuration at Rytov variance >= 1.0. At 0.2, eight modes on four carriers (twice the
spectral efficiency of WDM) have outage 6.8e-4, and concealment gives a failed-slot rate of 2.7e-5 over the eleven scenes
(`eval_layered_v2a_r1_T8R11W4_0.2w.json`).

Base layer with 6, 12 and 24 feature channels (garden, `*_v2a_r1_b6*`, `*_v2a_r1_b24*`): concealed PSNR in the T8R8 outage slots
25.91, 26.00 and 26.18 dB; one copy of positions and base layer 3.83, 5.04 and 7.46 MB.

HAC (official code, `hac/results/`): two to three times smaller on the indoor scenes and better without loss on three of five scenes, but in the same outage slots the layered
codec renders 2.70 dB above HAC with interpolation on average (5 scenes).

Latency on an idle NVIDIA L40S with the final attribute sizes (`codec/results/bench_l40s_idle_final.json`), 20,000 lost anchors:
gathering 0.36 + concealment (FP16, compiled) 3.92 + decoding 1.70 + rendering 2.16 = 8.14 ms.

## 6. Paper-to-script manifest

Every table and every number of the paper comes from the scripts and files below (S = supplementary material).

| Paper item | Script | Data |
|---|---|---|
| Configurations, outage vs SNR, failed streams, anchor pairs | `phy_sim/run_modeset.py` | `phy_sim/results/modeset_<s2>.json`, `traces_<s2>.npz` |
| Small-ball exponents (test of D2) | `phy_sim/run_smallball.py` | `smallball_<s2>.json` |
| Hardware tolerance (RIN x sorter loss) | `phy_sim/run_tolerance.py` | `tolerance_<s2>.json` |
| Model sensitivity | `phy_sim/run_phy_results.py --stages sens` | `sens_<s2>.json` |
| Receiver impairments (crosstalk, CSI, exact BER, FEC threshold) | `phy_sim/run_sens2.py` | `sens2_<s2>.json` |
| Zernike truncation (S) and extrapolation to J -> inf | `phy_sim/run_jconv.py`, `phy_sim/jconv_extrap.py` | `jconv_<s2>.json` |
| Gamma-Gamma surrogates (S), power allocation | `phy_sim/run_phy_results.py`, `phy_sim/run_alloc.py` | `oper_<s2>.json`, `alloc_T8R8_2.5.json` |
| Temporal Model B (S) | `phy_sim/run_traces_corr.py` | `results/correlated/` |
| Common carrier, wavelength reuse | `phy_sim/run_beat.py`, `phy_sim/run_reuse.py`, `phy_sim/run_reuse_curve.py` | `beat_<s2>.json`, `modeset_0.2w.json`, `traces_0.2w.npz`, `reuse_curve.json` |
| WDM reference | `phy_sim/wdm_reference.py` | printed |
| Split-step consistency | `phy_sim/splitstep/` | `phy_sim/splitstep/results_<s2>.json`, `patterns_tt_2.5.npz` |
| Reference 3DGS (S), codec quality and rate, main table, by K | `codec/train_3dgs.py`, `codec/train_layered.py`, `codec/eval_layered.py` via `codec/queue_exp.sh jobs_*.txt` | `codec/results/<scene>/metrics_*.json`, `eval_layered_*.json` |
| K beyond training (K = 1..7 of 8 forced) | `codec/run_kstress.sh` (`eval_layered.py --kforce`) | `eval_layered_v2a_r1_kstress_T8R8_2.5.json` |
| Seed spread | `codec/queue_exp.sh jobs_seeds.txt jobs_seeds2.txt` | `*_v2a_r1_sd1*`, `*_v2a_r1_sd2*` |
| Base-layer width | `codec/queue_exp.sh jobs_basedim.txt` | `*_v2a_r1_b6*`, `*_v2a_r1_b24*` |
| Mapping policies (S), rate weights (S) | `codec/eval_layered.py`, `codec/queue_exp.sh jobs_garden_v2.txt` | `eval_layered_v2a_r*` |
| Frame-level evaluation with edge cache | `codec/eval_frames.py`, `run_frames*.sh`, `run_fr10.sh` | `frames_*.json` |
| Reuse link codec evaluation | `codec/run_reuse_eval.sh` | `eval_layered_v2a_r1_T*W4_0.2w.json` |
| HAC under loss, HAC at equal transmitted bits | `hac/` | `hac/results/` |
| Latency | `codec/bench_latency.py`, `codec/bench_final.sh` | `codec/results/bench_l40s_idle_final.json` |
| Failed-slot rate over the RIN x loss grid | `codec/tol_service.py` | `phy_sim/results/tol_service_2.5.json` |
| Scheduler under time-varying turbulence | `codec/sched_tv.py` | `phy_sim/results/sched_tv.json` |
| Cold start: concealment, in-frame ARQ, 2x replication | `codec/arq_cold.py` | `phy_sim/results/arq_cold.json` (patterns of `phy_sim/results/correlated/patterns/`) |
| HAC with an MDS code at equal transmitted bits | `codec/hac_eqbits.py` | `codec/results/hac_eqbits.json` |
| Channel variation inside a slot, pilot aging | `phy_sim/run_aging.py` | `phy_sim/results/aging_<tau>ms.json` |
| Combined impairment budget | `phy_sim/run_budget.py` | `phy_sim/results/budget_2.5.json` |
| WDM with PAM-L at the spectral efficiency of reuse | `phy_sim/wdm_pam.py` | `phy_sim/results/wdm_pam.json` |

## 7. Status

All results of the paper are in this repository. Training ran on a shared GPU, so the wall-clock times in the logs are not
representative; the latency measurements were taken with all other jobs of this project paused.

## 8. License and citation

MIT License (see `LICENSE`). If you use the code or the traces, please cite the paper (reference to be added on
publication).
