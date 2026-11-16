#!/bin/bash
# Codec evaluation on the multi-beam FSO traces (second channel): one configuration per worker.
cd "$HOME/NamNX'sPaper/Q1_2nd/codec"
export CUDA_HOME=/usr/local/cuda-12.6 PATH=/usr/local/cuda-12.6/bin:$PATH TORCH_CUDA_ARCH_LIST=8.9
PY=~/.venvs/q1_2nd/bin/python; L=results/logs; G=$L/guard_review3.log; c=$1
for s in garden stump room counter bonsai truck train playroom bicycle kitchen drjohnson; do
  out=results/$s/eval_layered_v2a_r1_${c}_mb.json
  for try in 1 2 3 4; do
    [ -f $out ] && break
    until (( $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1) >= 7000 )) && (( $(awk "/MemAvailable/{printf \"%d\", \$2/1048576}" /proc/meminfo) >= 6 )); do sleep 60; done
    $PY -u eval_layered.py --scene $s --tag _v2a_r1 --cfg $c --s2 mb --n-pat 6 --policies interleaved > $L/evl_${s}_mb_$c.log 2>&1 && break
    sleep 300
  done
  echo "$(date -Is) mb $c $s $( [ -f $out ] && echo ok || echo FAIL)" >> $G
done
