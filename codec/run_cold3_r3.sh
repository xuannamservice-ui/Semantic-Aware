#!/bin/bash
# Concealing schemes at a cold start with the deployment variant on every scene and the concealment neighbours restricted
# to the anchors of the view (--view-nbr, suffix _r3). Waits for the edge models and the tau_c = 10 ms runs.
# usage: ./run_cold3_r3.sh <scene> ...
cd "$HOME/NamNX'sPaper/Q1_2nd/codec"
export CUDA_HOME=/usr/local/cuda-12.6 PATH=/usr/local/cuda-12.6/bin:$PATH TORCH_CUDA_ARCH_LIST=8.9
source ./gpu_lock.sh; PY=~/.venvs/q1_2nd/bin/python; L=results/logs; G=$L/guard_review3.log
for s in "$@"; do
  until [ -f results/$s/edge_v2a_r1.pt ]; do sleep 60; done
  for cfg in T8R8 T6R11; do na=4; [ $cfg = T6R11 ] && na=2
    for tau in 2.5 10; do
      out=results/$s/cold3_v2a_r1_${cfg}_tau${tau}_r3.json
      [ -f $out ] || start_locked 5000 4 $L/cold3v_${s}_${cfg}_${tau}.log $PY -u eval_cold3.py --scene $s --cfg $cfg --tau $tau --n-anchor $na --n-ev 3 --n-mds 2 --phy phy --schemes conceal,rep2_conceal,mds_conceal,arq_conceal --suffix _r3 --view-nbr
      echo "$(date -Is) cold3v $cfg tau$tau $s $( [ -f $out ] && echo ok || echo FAIL)" >> $G
    done
  done
done
echo "$(date -Is) cold3v worker $1 done" >> $G
