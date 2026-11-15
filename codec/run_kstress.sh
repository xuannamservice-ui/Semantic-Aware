#!/bin/bash
# Stress test beyond the training distribution: K = 1..7 of the 8 streams fail (random subsets), interleaved mapping.
cd "$HOME/NamNX'sPaper/Q1_2nd/codec"
export CUDA_HOME=/usr/local/cuda-12.6 PATH=/usr/local/cuda-12.6/bin:$PATH TORCH_CUDA_ARCH_LIST=8.9
PY=~/.venvs/q1_2nd/bin/python; L=results/logs; G=$L/guard_review3.log
for s in garden stump room counter bonsai truck train playroom bicycle kitchen drjohnson; do
  out=results/$s/eval_layered_v2a_r1_kstress_T8R8_2.5.json
  for try in 1 2 3 4; do
    [ -f $out ] && break
    until (( $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1) >= 7000 )) && (( $(awk "/MemAvailable/{printf \"%d\", \$2/1048576}" /proc/meminfo) >= 6 )); do sleep 60; done
    $PY -u eval_layered.py --scene $s --tag _v2a_r1 --cfg T8R8 --s2 2.5 --n-pat 2 --kforce 1,2,3,4,5,6,7 --policies interleaved --suffix _kstress > $L/evl_${s}_kstress.log 2>&1 && break
    sleep 300
  done
  echo "$(date -Is) kstress $s $( [ -f $out ] && echo ok || echo FAIL)" >> $G
done
