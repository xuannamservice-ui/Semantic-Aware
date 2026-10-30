# Semantic-Aware Generative Error Concealment for Resilient Real-Time 3DGS Streaming over Turbulent FSO Backhaul

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![IEEE Access](https://img.shields.io/badge/Target-IEEE%20Access-orange.svg)](https://ieeeaccess.ieee.org/)
[![Reproducibility](https://img.shields.io/badge/Reproducibility-10%5E6%20Slots-success.svg)](#reproducibility-pipeline)

This repository contains the official simulation source code, experimental datasets, and plotting routines supporting the paper:

> **Semantic-Aware Generative Error Concealment for Resilient Real-Time 3D Gaussian Splatting Streaming over Turbulent FSO Backhaul**  
> *Pham Vu Minh Tu, Nguyen Xuan Nam, Vu Van Thuong, Tien Vu Huu*  
> Creative Technologies and Simulation Lab, Posts and Telecommunications Institute of Technology, Hanoi, Vietnam.

---

## 🔬 Key Scientific Modules & Limitations Addressed

This release provides comprehensive numerical evidence addressing the boundary conditions and operational limitations:

1. **Temporal Correlated Fading & Burst Erasures (`src/temporal_correlated_fading.py`)**
   - Simulates $1{,}000{,}000$ consecutive transmission slots under Taylor's frozen-flow atmospheric turbulence hypothesis ($\tau_c = 2.5\text{ ms}$, $\rho = 0.8048$).
   - Quantifies the burst erasure length distribution $L_{burst}$ and confirms that Semantic Anchor replication on the uncorrelated mode pair $(-2, +1)$ suppresses burst loss events of length $L \ge 4$ to **0%**, stabilizing the Lyapunov virtual queue ($Z(t) \le 35.4 \ll 100$).

2. **Hardware Constraints & Laser RIN Sensitivity (`src/hardware_rin_insertion_loss_sweep.py`)**
   - Performs a 210-point parameter sweep across laser Relative Intensity Noise (RIN $\in [-165, -135]\text{ dB/Hz}$) and mode-sorter excess insertion loss ($\Delta\eta \in [0, 4.5]\text{ dB}$).
   - Identifies the operating boundary where mode throttling ($M=6, N_r=11$) sustains the $10^{-4}$ URLLC target, and proves that generative error concealment is indispensable when using square mode sorters.

3. **Dual-Regime Dynamic Precision Adaptation (`src/dynamic_precision_relaxation.py`)**
   - Evaluates the Block Coordinate Descent (BCD) Lyapunov scheduler across 10 latency deadline horizons ($D_{max} \in [0.75, 3.50]\text{ ms}$).
   - Demonstrates dual-regime operation: static 1.58-bit ternary quantization ($Q=1.0$) under extreme sub-millisecond URLLC ($0.78\text{ ms}$), and dynamic precision modulation under standard interactive XR budgets ($1.5 - 3.0\text{ ms}$), achieving up to $+3.4\text{ dB}$ PSNR gain.

---

## 📁 Repository Structure

```
.
├── LICENSE                                # MIT Open-Source License
├── README.md                              # This documentation
├── requirements.txt                       # Python dependencies
├── src/                                   # Source code
│   ├── temporal_correlated_fading.py      # Exp 1: 1M-slot temporal fading simulation
│   ├── hardware_rin_insertion_loss_sweep.py# Exp 2: 2D hardware sensitivity sweep
│   ├── dynamic_precision_relaxation.py    # Exp 3: Precision adaptation trade-off
│   ├── plot_results.py                    # 300-DPI publication figure generator
│   └── run_all_experiments.sh             # Master execution script
├── data/                                  # Raw JSON and CSV results
│   ├── exp1_temporal_correlated_1m_results.json
│   ├── exp2_hardware_rin_sensitivity.json
│   ├── exp2_hardware_rin_sensitivity.csv
│   ├── exp3_dynamic_precision_pareto.json
│   └── exp3_dynamic_precision_pareto.csv
└── figures/                               # Generated 300 DPI publication plots
    ├── fig_exp1_burst_erasures_and_stability.png
    ├── fig_exp2_hardware_rin_envelope.png
    └── fig_exp3_dynamic_precision_pareto.png
```

---

## 🚀 Quickstart & Reproduction

### Prerequisites
* Python 3.9+ (tested up to Python 3.14 on Ubuntu 24.04 LTS / macOS Sonoma)
* Standard libraries: `numpy`, `scipy`, `matplotlib`

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the complete pipeline (executes simulations & generates figures)
bash src/run_all_experiments.sh
```

All figures will be populated in `figures/` and raw datasets in `data/`.

---

## 📜 Citation & License

This project is licensed under the [MIT License](LICENSE).
If you use this codebase or data in your research, please cite our IEEE Access paper:

```bibtex
@article{tu2026semantic3dgs,
  author={Tu, Pham Vu Minh and Nam, Nguyen Xuan and Thuong, Vu Van and Huu, Tien Vu},
  journal={IEEE Access}, 
  title={Semantic-Aware Generative Error Concealment for Resilient Real-Time 3D Gaussian Splatting Streaming over Turbulent FSO Backhaul}, 
  year={2026},
  volume={14},
  doi={10.1109/ACCESS.2026.xxxxxxx}
}
```
