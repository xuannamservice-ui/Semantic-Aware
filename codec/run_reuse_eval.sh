#!/bin/bash
# Codec evaluation on the wavelength-reuse traces (sigma_R^2 = 0.2, 4 carriers): one configuration per worker.
cd "$HOME/NamNX'sPaper/Q1_2nd/codec"
export CUDA_HOME=/usr/local/cuda-12.6 PATH=/usr/local/cuda-12.6/bin:$PATH TORCH_CUDA_ARCH_LIST=8.9
PY=~/.venvs/q1_2nd/bin/python; L=results/logs; G=$L/guard_review2.log; c=$1
gpu_free_mb() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
mem_avail() { awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo; }
for s in garden stump room counter bonsai truck train playroom bicycle kitchen drjohnson; do
  out=results/$s/eval_layered_v2a_r1_${c}_0.2w.json
  for try in 1 2 3 4 5 6; do
    [ -f $out ] && break
    until (( $(gpu_free_mb) >= 7000 )) && (( $(mem_avail) >= 7 )); do sleep 60; done
    $PY -u eval_layered.py --scene $s --tag _v2a_r1 --cfg $c --s2 0.2w --n-pat 6 > $L/evl_${s}_reuse_$c.log 2>&1; rc=$?
    [ $rc -eq 0 ] && break
    (( rc >= 128 )) || grep -qiE "out of memory|OutOfMemory" $L/evl_${s}_reuse_$c.log || break
    sleep 300
  done
  echo "$(date -Is) reuse eval $c $s $( [ -f $out ] && echo ok || echo FAIL)" >> $G
done
