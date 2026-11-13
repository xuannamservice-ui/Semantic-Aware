#!/bin/bash
# Latency on an otherwise idle GPU: pause (SIGSTOP) every Q1_2nd GPU job, run the benchmark, resume (SIGCONT).
# Paused jobs keep their GPU memory but use no compute. Resumes the jobs even if the benchmark fails.
cd "$(dirname "$0")"
export CUDA_HOME=/usr/local/cuda-12.6 PATH=/usr/local/cuda-12.6/bin:$PATH TORCH_CUDA_ARCH_LIST=8.9
PY=~/.venvs/q1_2nd/bin/python
pids=$(pgrep -u "$USER" -f "q1_2nd/bin/python -u (train_layered|train_lite|eval_layered|eval_trajectory)")
echo "pausing: $pids"; kill -STOP $pids
trap 'kill -CONT $pids; echo resumed' EXIT
sleep 5
nvidia-smi --query-gpu=utilization.gpu,memory.used,clocks.sm --format=csv,noheader
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
$PY bench_latency.py --out results/bench_l40s_idle.json --lost 1000,5000,10000,20000,30000 --A-vis 110000 --anchors 250000 --n-warm 50 --n-rep 2000 --compile 1 2>&1 | grep -v -i warn | grep -E "^[0-9]|decode|Error"
