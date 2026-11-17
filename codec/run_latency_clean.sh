#!/bin/bash
# One-session latency measurement on an otherwise idle GPU: synthetic stages (bench_latency.py), then the trained models
# (bench_trained.py) referred to these synthetic stages; the GPU process list is logged before and after each step.
cd "$HOME/NamNX'sPaper/Q1_2nd/codec"
export CUDA_HOME=/usr/local/cuda-12.6 PATH=/usr/local/cuda-12.6/bin:$PATH TORCH_CUDA_ARCH_LIST=8.9
PY=~/.venvs/q1_2nd/bin/python; L=results/logs/latency_clean.log
snap() { echo "== $1 $(date -Is)"; nvidia-smi --query-gpu=clocks.sm,power.draw,temperature.gpu,utilization.gpu,memory.used --format=csv,noheader; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader; uptime; } >> $L
: > $L
until [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 30; done
snap before-synthetic
$PY bench_latency.py --out results/bench_l40s_clean.json --lost 1000,5000,10000,20000,30000 --A-vis 110000 --anchors 250000 --n-warm 50 --n-rep 1000 --compile 1 >> $L 2>&1
snap after-synthetic
$PY -u bench_trained.py --idle results/bench_l40s_clean.json >> $L 2>&1
snap after-trained
echo "done" >> $L
