#!/bin/bash
# Cold start at tau_c = 10 ms on the six scenes not yet run (all schemes, then replication/MDS with the _r2 suffix).
cd "$HOME/NamNX'sPaper/Q1_2nd/codec"
export CUDA_HOME=/usr/local/cuda-12.6 PATH=/usr/local/cuda-12.6/bin:$PATH TORCH_CUDA_ARCH_LIST=8.9
source ./gpu_lock.sh; PY=~/.venvs/q1_2nd/bin/python; L=results/logs; G=$L/guard_review3.log
SIX="kitchen truck train playroom bicycle drjohnson"
for cfg in T8R8 T6R11; do na=4; [ $cfg = T6R11 ] && na=2
  for s in $SIX; do
    out=results/$s/cold3_v2a_r1_${cfg}_tau10.json
    [ -f $out ] || start_locked 5000 4 $L/cold3_${s}_${cfg}_10.log $PY -u eval_cold3.py --scene $s --cfg $cfg --tau 10 --n-anchor $na --n-ev 3 --n-mds 2 --phy phy
    out2=results/$s/cold3_v2a_r1_${cfg}_tau10_r2.json
    [ -f $out2 ] || start_locked 5000 4 $L/cold3r_${s}_${cfg}_10.log $PY -u eval_cold3.py --scene $s --cfg $cfg --tau 10 --n-anchor $na --n-ev 3 --n-mds 2 --phy phy --schemes rep2,rep2_conceal,mds,mds_conceal --suffix _r2
    echo "$(date -Is) cold3ext $cfg $s $( [ -f $out ] && [ -f $out2 ] && echo ok || echo FAIL)" >> $G
  done
done
echo "$(date -Is) cold3ext done" >> $G
