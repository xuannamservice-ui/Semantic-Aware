# Semantic-Aware Generative Error Concealment for Resilient Real-Time 3D Gaussian Splatting Streaming over Turbulent FSO Backhaul

Code, simulation data and measured results for the paper of the same title
(P. V. M. Tu, N. X. Nam, V. V. Thuong, T. V. Huu, Creative Technologies and Simulation Lab, Posts and
Telecommunications Institute of Technology, Hanoi).

The repository has two independent parts that meet in one interface, the **per-slot failure pattern** of the
optical link (which of the parallel streams failed in a slot):

| Part | Folder | What it does | Hardware |
|---|---|---|---|
| Optical link | [`phy_sim/`](phy_sim/) | Wave-optics Monte Carlo of an intensity-modulated OAM free-space link with a mode sorter and linear MMSE equalization; writes outage statistics and **failure-pattern traces** | CPU only |
| Codec | [`codec/`](codec/) | Concealment-aware layered 3D Gaussian Splatting codec and generative concealment network; trains on Mip-NeRF 360, Tanks and Temples and Deep Blending scenes and evaluates the rendering quality under the failure traces | NVIDIA GPU |

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
  queue_*.sh, run_*.sh        job queues used on the shared GPU server (memory guards included)
  results/                    per-scene metrics and evaluations (JSON); see section 4
  archive/                    first-generation (non-layered) codec scripts kept for reference; not used for the paper
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
./launch_all.sh                                               # the full set as published
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
python bench_latency.py --out results/bench.json --lost 1000,5000,10000,20000,30000
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

Codec, garden (Mip-NeRF 360), final joint design: loss-free 26.71 dB PSNR / 0.186 LPIPS (reference 3DGS 26.32 / 0.222),
18.6 MB; in outage slots of the 8-mode/8-port trace: 26.02 dB with concealment, 19.13 dB without, 23.52 dB with
neighbour interpolation (`results/garden/eval_layered_v2a_r1_T8R8_2.5.json`).

Latency on an idle NVIDIA L40S, 10,000 lost anchors: LiteConcealNet FP16 (compiled) 1.77 ms, EdgeConcealNet FP16
1.06 ms; anchor-to-Gaussian decoding of 110k visible anchors 1.62 ms; rendering at 1297 x 840 2.07 ms.

## 6. Status

The trainings and frame-level evaluations of the remaining scenes, the HAC comparison on the same protocol and the
multi-screen split-step validation of the channel model are running; their results are added to `codec/results/` and
`phy_sim/results/` as they finish.

## 7. License and citation

MIT License (see `LICENSE`). If you use the code or the traces, please cite the paper (reference to be added on
publication).
