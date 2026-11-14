#!/bin/bash
# More chains (50) for the T8R8 frame-level evaluations, the only configuration with frequent misses; the first 10
# chains are identical to the 10-chain runs (same seeds). Starts after the 10-chain queues have finished.
cd "$(dirname "$0")"
export CUDA_HOME=/usr/local/cuda-12.6 PATH=/usr/local/cuda-12.6/bin:$PATH TORCH_CUDA_ARCH_LIST=8.9
PY=~/.venvs/q1_2nd/bin/python; L=results/logs
gpu_free() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | awk '{printf "%d", $1/1024}'; }
mem_avail() { awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo; }
while pgrep -f run_frames2.sh > /dev/null; do sleep 120; done
for s in garden stump room counter bonsai truck train playroom bicycle kitchen drjohnson; do
  [ -f results/$s/metrics_layered_v2a_r1.json ] || continue
  for tau in 10 2.5; do
    for try in 1 2 3 4 5 6; do
      until (( $(gpu_free) >= 8 )) && (( $(mem_avail) >= 8 )); do sleep 60; done
      $PY -u eval_frames.py --scene $s --tag _v2a_r1 --cfg T8R8 --tau $tau --n-anchor 4 --chains 50 > $L/frames50_${s}_T8R8_$tau.log 2>&1; rc=$?
      [ $rc -eq 0 ] && break
      (( rc >= 128 )) || grep -qiE "out of memory|OutOfMemory" $L/frames50_${s}_T8R8_$tau.log || break
      sleep 300
    done
  done
done
echo "$(date -Is) frames_more done" >> $L/guard_exp.log
