#!/bin/bash
# Outage-slot evaluation on the failure patterns of the split-step wave-optics channel (phy_ss/, s2 label "ss2.5").
cd "$(dirname "$0")"
export CUDA_HOME=/usr/local/cuda-12.6 PATH=/usr/local/cuda-12.6/bin:$PATH TORCH_CUDA_ARCH_LIST=8.9
PY=~/.venvs/q1_2nd/bin/python; L=results/logs
gpu_free() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | awk '{printf "%d", $1/1024}'; }
mem_avail() { awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo; }
for s in garden stump counter truck playroom bonsai train room; do
  for c in T8R8 T6R8; do
    f=results/$s/eval_layered_v2a_r1_${c}_ss2.5.json
    for try in 1 2 3 4 5 6; do
      [ -f $f ] && break
      until (( $(gpu_free) >= 8 )) && (( $(mem_avail) >= 8 )); do sleep 60; done
      $PY -u eval_layered.py --scene $s --tag _v2a_r1 --cfg $c --s2 ss2.5 --phy phy_ss --n-pat 6 > $L/evl_ss_${s}_$c.log 2>&1; rc=$?
      [ $rc -eq 0 ] && break
      (( rc >= 128 )) || grep -qiE "out of memory|OutOfMemory" $L/evl_ss_${s}_$c.log || break
      sleep 300
    done
  done
done
echo "$(date -Is) ss eval done" >> $L/guard_exp.log
