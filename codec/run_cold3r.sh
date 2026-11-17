#!/bin/bash
# Replication and MDS at a cold start with the anchor modes used after slot S_A (eval_cold3.py --schemes ... --suffix _r2).
cd "$HOME/NamNX'sPaper/Q1_2nd/codec"
export CUDA_HOME=/usr/local/cuda-12.6 PATH=/usr/local/cuda-12.6/bin:$PATH TORCH_CUDA_ARCH_LIST=8.9
source ./gpu_lock.sh; PY=~/.venvs/q1_2nd/bin/python; L=results/logs; G=$L/guard_review3.log
while pgrep -f "[r]un_cold3.sh" > /dev/null; do sleep 30; done
ALL="kitchen garden stump room counter bonsai truck train playroom bicycle drjohnson"; FIVE="garden stump room counter bonsai"
run() { cfg=$1; tau=$2; na=$3; shift 3
  for s in "$@"; do
    out=results/$s/cold3_v2a_r1_${cfg}_tau${tau}_r2.json
    [ -f $out ] || start_locked 5000 5 $L/cold3r_${s}_${cfg}_${tau}.log $PY -u eval_cold3.py --scene $s --cfg $cfg --tau $tau --n-anchor $na --n-ev 3 --n-mds 2 --phy phy --schemes rep2,rep2_conceal,mds,mds_conceal --suffix _r2
    echo "$(date -Is) cold3r $cfg tau$tau $s $( [ -f $out ] && echo ok || echo FAIL)" >> $G
  done
}
run T8R8 2.5 4 $ALL; run T6R11 2.5 2 $ALL; run T8R8 10 4 $FIVE; run T6R11 10 2 $FIVE
touch /tmp/q1_cold3r_done; echo "$(date -Is) cold3r done" >> $G
