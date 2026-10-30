# Semantic-Aware Generative Error Concealment for Resilient Real-Time 3DGS Streaming over Turbulent FSO Backhaul

This repository contains reproducible numerical verification code supporting the paper:

> **Semantic-Aware Generative Error Concealment for Resilient Real-Time 3D Gaussian Splatting Streaming over Turbulent FSO Backhaul**  
> *Pham Vu Minh Tu, Nguyen Xuan Nam, Vu Van Thuong, Tien Vu Huu*  
> Creative Technologies and Simulation Lab, Posts and Telecommunications Institute of Technology, Hanoi, Vietnam.  
> Target: *IEEE Access*

---

## Scope and Contents

This release provides the following standalone verification modules:

1. **Temporal Correlated Fading Surrogate Model (`src/surrogate_recalibrated.py`)**
   - Implements the bivariate Gauss-Markov temporal correlation model described in Section VII-A of the manuscript.
   - Simulates $10^7$ consecutive transmission slots with channel coherence time $\tau_c = 2.5\text{ ms}$, slot duration $T_f = 1.0\text{ ms}$, and one-slot correlation $\rho = \exp(-(T_f/\tau_c)^{5/3}) \approx 0.8048$.
   - Accurately recalibrated to the physical-layer Monte Carlo loss statistics under strong turbulence ($\sigma_R^2 = 2.5$, square sorter $M = N_r = 8$): single-mode outage $P(+1) = 0.018$, joint anchor-pair loss $P((-2, +1)) = 1.75 \times 10^{-3}$, cross-correlation $r_{cross} = 0.36$.
   - Computes burst erasure distributions and evaluates Lyapunov virtual queue drift rates.

2. **Data-Driven Publication Plotting Routines (`src/plot_from_simulation_data.py`)**
   - Clean plotting script that generates publication figures directly from raw Monte Carlo and GPU measurement CSV files (`data/`).
   - Automatically skips curves when raw measurement files are empty, ensuring no artificial or ungrounded curves are rendered.

---

## Implementation & Reproduction Note

As stated in the manuscript:
- The physical-layer Monte Carlo simulations ($10^7$ independent Zernike phase-screen realizations, aperture scintillation integration, and linear MMSE equalization) and the 24-layer 605M-parameter BitLinear 1.58-bit neural network training were profiled on NVIDIA L40S 48GB GPU hardware.
- Tabulated channel statistics, rate-distortion envelopes, and hardware latencies are reported in the manuscript tables. The surrogate script in this repository reproduces the temporal fading dynamics and burst statistics derived from those operating points.

---

## Directory Structure

```
.
├── LICENSE                          # MIT Open-Source License
├── README.md                        # Documentation
├── requirements.txt                 # Python dependencies
├── src/
│   ├── surrogate_recalibrated.py    # 10^7 slot Gauss-Markov burst simulation
│   └── plot_from_simulation_data.py # Data-driven plotting script
└── data/                            # Measurement templates
    ├── mc_outage_vs_snr.csv
    ├── gpu_latency_vs_Q.csv
    └── zernike_truncation.csv
```

---

## Quickstart

### Prerequisites
* Python 3.9+
* Required libraries: `numpy`, `scipy`, `matplotlib`

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the temporal correlated surrogate simulation (10^7 slots)
python src/surrogate_recalibrated.py
```

---

## Citation & License

This project is open-source under the [MIT License](LICENSE).  
If you reference this work, please cite:

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
