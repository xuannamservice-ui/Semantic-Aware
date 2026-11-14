#!/bin/bash
# Frame-level stress tests: faster camera (stride 2/4/8 over the capture path) for T8R8, and the T8R11 configuration.
cd "$(dirname "$0")"
export CUDA_HOME=/usr/local/cuda-12.6 PATH=/usr/local/cuda-12.6/bin:$PATH TORCH_CUDA_ARCH_LIST=8.9
PY=~/.venvs/q1_2nd/bin/python; L=results/logs
gpu_free() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | awk '{printf "%d", $1/1024}'; }
mem_avail() { awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo; }
run() { log=$1; shift
  for try in 1 2 3 4 5 6; do
    until (( $(gpu_free) >= 8 )) && (( $(mem_avail) >= 8 )); do sleep 60; done
    $PY -u eval_frames.py "$@" > $L/$log.log 2>&1; rc=$?
    [ $rc -eq 0 ] && return 0
    (( rc >= 128 )) || grep -qiE "out of memory|OutOfMemory" $L/$log.log || return 1
    sleep 300
  done; }
for st in 4 8 2; do for s in garden room truck kitchen; do
  [ -f results/$s/frames_v2a_r1_T8R8_tau10_eps0.166667_stride$st.json ] || run frames_${s}_T8R8_10_s$st --scene $s --tag _v2a_r1 --cfg T8R8 --tau 10 --n-anchor 4 --chains 10 --stride $st
done; done
for s in garden room truck kitchen; do
  [ -f results/$s/frames_v2a_r1_T8R11_tau10_eps0.166667.json ] || run frames_${s}_T8R11_10 --scene $s --tag _v2a_r1 --cfg T8R11 --tau 10 --n-anchor 2 --chains 10
done
echo "$(date -Is) stress done" >> $L/guard_exp.log
