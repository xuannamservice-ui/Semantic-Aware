#!/bin/bash
# Decoding and rasterization of the trained models on their test views against the synthetic benchmark (bench_trained.py).
cd "$HOME/NamNX'sPaper/Q1_2nd/codec"
export CUDA_HOME=/usr/local/cuda-12.6 PATH=/usr/local/cuda-12.6/bin:$PATH TORCH_CUDA_ARCH_LIST=8.9
source ./gpu_lock.sh; PY=~/.venvs/q1_2nd/bin/python; L=results/logs; G=$L/guard_review3.log
start_locked 5000 8 $L/bench_trained.log $PY -u bench_trained.py
echo "$(date -Is) bench_trained $( [ -f results/bench_trained.json ] && echo ok || echo FAIL)" >> $G
