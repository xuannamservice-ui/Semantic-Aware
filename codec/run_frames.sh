#!/bin/bash
# Frame-level evaluations (main scenario) for the given scenes once their final model and the pattern traces exist.
# Configurations T8R8 / T6R8 / T6R11 at tau_c = 10 ms and T8R8 at 2.5 ms; eps = 1/6; base layer on 2 modes
# (4 modes for T8R8). One evaluation at a time; waits for >= 8 GB free GPU memory and retries after OOM.
cd "$(dirname "$0")"
export CUDA_HOME=/usr/local/cuda-12.6 PATH=/usr/local/cuda-12.6/bin:$PATH TORCH_CUDA_ARCH_LIST=8.9
PY=~/.venvs/q1_2nd/bin/python; L=results/logs
gpu_free() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | awk '{printf "%d", $1/1024}'; }
until [ -f phy/patterns/READY ]; do sleep 120; done
for s in "$@"; do
  until [ -f results/$s/metrics_layered_v2a_r1.json ]; do sleep 120; done
  for job in "T8R8 10 4" "T6R8 10 2" "T6R11 10 2" "T8R8 2.5 4"; do
    set -- $job; out=results/$s/frames_v2a_r1_${1}_tau${2}_eps0.166667.json
    for try in 1 2 3 4 5 6; do
      [ -f $out ] && break
      until (( $(gpu_free) >= 8 )); do sleep 60; done
      $PY -u eval_frames.py --scene $s --tag _v2a_r1 --cfg $1 --tau $2 --n-anchor $3 --chains 10 > $L/frames_${s}_$1_$2.log 2>&1 && break
      grep -qiE "out of memory|OutOfMemory" $L/frames_${s}_$1_$2.log || break
      sleep 300
    done
  done
done
