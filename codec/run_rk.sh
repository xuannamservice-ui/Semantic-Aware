#!/bin/bash
# Training-free regression-kriging baseline (ridge regression from the base to the enhancement layer on the received
# anchors of the view, plus IDW of the neighbours' residuals) on the outage slots of the T8R8 trace, with IDW and the
# network on the same draws as the main evaluation (seed 0, six patterns per view). usage: ./run_rk.sh <scene> ...
cd "$HOME/NamNX'sPaper/Q1_2nd/codec"
export CUDA_HOME=/usr/local/cuda-12.6 PATH=/usr/local/cuda-12.6/bin:$PATH TORCH_CUDA_ARCH_LIST=8.9
source ./gpu_lock.sh; PY=~/.venvs/q1_2nd/bin/python; L=results/logs; G=$L/guard_review3.log
for s in "$@"; do
  out=results/$s/eval_layered_v2a_r1_rk_T8R8_2.5.json
  [ -f $out ] || start_locked 6000 5 $L/rk_${s}.log $PY -u eval_layered.py --scene $s --tag _v2a_r1 --cfg T8R8 --s2 2.5 --n-pat 6 --rk --methods idw,rk,gen_fp --policies interleaved,importance --suffix _rk
  echo "$(date -Is) rk $s $( [ -f $out ] && echo ok || echo FAIL)" >> $G
done
echo "$(date -Is) rk worker $1 done" >> $G
