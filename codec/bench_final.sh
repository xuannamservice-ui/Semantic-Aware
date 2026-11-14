#!/bin/bash
# Final latency run with the attribute sizes of the final codec (18 base, 66 enhancement values, 48 feature channels).
# Waits until the rate-matched garden training has finished, then pauses (SIGSTOP) every GPU job of this project,
# including the HAC runs and the frame evaluations, measures, and resumes them (SIGCONT) even on failure.
cd "$(dirname "$0")"
export CUDA_HOME=/usr/local/cuda-12.6 PATH=/usr/local/cuda-12.6/bin:$PATH TORCH_CUDA_ARCH_LIST=8.9
PY=~/.venvs/q1_2nd/bin/python
until [ -f results/garden/metrics_layered_v2s_r0.json ]; do sleep 120; done
pids=$(pgrep -u "$USER" -f "(q1_2nd/bin/python -u (train_layered|train_lite|eval_layered|eval_frames|eval_trajectory))|(python (train.py|hac_trace_eval.py))" | tr "\n" " ")
echo "$(date -Is) pausing: $pids"; kill -STOP $pids
trap 'kill -CONT $pids; echo "$(date -Is) resumed"' EXIT
sleep 10
nvidia-smi --query-gpu=utilization.gpu,memory.used,clocks.sm --format=csv,noheader
$PY bench_latency.py --out results/bench_l40s_idle_final.json --lost 1000,5000,10000,20000,30000 --A-vis 110000 --anchors 250000 --n-warm 50 --n-rep 2000 --compile 1 2>&1 | grep -v -i warn | tail -30
