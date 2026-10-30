#!/bin/bash
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

echo "=========================================================="
echo " 3DGS-OAM-FSO Large-Scale Reproducibility Pipeline"
echo " Host: $(hostname)"
echo " CPU Threads: $(nproc)"
echo " Date: $(date)"
echo "=========================================================="

mkdir -p "$ROOT_DIR/data" "$ROOT_DIR/figures" "$ROOT_DIR/results"

echo ""
echo ">>> [1/4] Running 1,000,000-Slot Correlated Fading & Burst Simulation..."
python3 src/temporal_correlated_fading.py

echo ""
echo ">>> [2/4] Running High-Resolution Hardware Constraints & RIN Sweep..."
python3 src/hardware_rin_insertion_loss_sweep.py

echo ""
echo ">>> [3/4] Running Multi-Horizon Dynamic Precision Adaptation..."
python3 src/dynamic_precision_relaxation.py

echo ""
echo ">>> [4/4] Generating Publication Figures (300 DPI)..."
python3 src/plot_results.py

echo ""
echo "=========================================================="
echo " Packaging Release Archive..."
tar -czf 3dgs_fso_urllc_reproducible_release_v1.0.tar.gz README.md LICENSE requirements.txt run_all_experiments.sh src/ data/ figures/
echo " Archive created: $ROOT_DIR/3dgs_fso_urllc_reproducible_release_v1.0.tar.gz"
echo " Size: $(du -h 3dgs_fso_urllc_reproducible_release_v1.0.tar.gz | cut -f1)"
echo " Pipeline executed successfully!"
echo "=========================================================="
