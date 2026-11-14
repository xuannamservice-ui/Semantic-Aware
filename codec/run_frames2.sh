#!/bin/bash
# As run_frames.sh, but also retries a run that was terminated by the RAM watchdog (exit status >= 128).
cd "$(dirname "$0")"
export CUDA_HOME=/usr/local/cuda-12.6 PATH=/usr/local/cuda-12.6/bin:$PATH TORCH_CUDA_ARCH_LIST=8.9
PY=~/.venvs/q1_2nd/bin/python; L=results/logs
gpu_free() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | awk '{printf "%d", $1/1024}'; }
mem_avail() { awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo; }
for s in "$@"; do
  until [ -f results/$s/metrics_layered_v2a_r1.json ]; do sleep 120; done
  for job in "T8R8 10 4" "T6R8 10 2" "T6R11 10 2" "T8R8 2.5 4"; do
    set -- $job; out=results/$s/frames_v2a_r1_${1}_tau${2}_eps0.166667.json
    for try in 1 2 3 4 5 6 7 8; do
      [ -f $out ] && break
      until (( $(gpu_free) >= 8 )) && (( $(mem_avail) >= 8 )); do sleep 60; done
      $PY -u eval_frames.py --scene $s --tag _v2a_r1 --cfg $1 --tau $2 --n-anchor $3 --chains 10 > $L/frames_${s}_$1_$2.log 2>&1; rc=$?
      [ $rc -eq 0 ] && break
      (( rc >= 128 )) || grep -qiE "out of memory|OutOfMemory" $L/frames_${s}_$1_$2.log || break
      sleep 300
    done
  done
done
echo "$(date -Is) frames $* done" >> $L/guard_exp.log
