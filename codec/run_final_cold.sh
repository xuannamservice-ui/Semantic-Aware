#!/bin/bash
# 1) grouped rANS (6 channels per substream) on all scenes; 2) the final, consistent cold-start run (all schemes, deployment
# variant on every scene, neighbours within the view, GPU entropy decoding of the measured 1.73 ms per 112,000 anchors).
cd "$HOME/NamNX'sPaper/Q1_2nd/codec"
export CUDA_HOME=/usr/local/cuda-12.6 PATH=/usr/local/cuda-12.6/bin:$PATH TORCH_CUDA_ARCH_LIST=8.9
source ./gpu_lock.sh; PY=~/.venvs/q1_2nd/bin/python; L=results/logs; G=$L/guard_review3.log
ALL="garden stump room counter bonsai truck train playroom bicycle kitchen drjohnson"
for s in $ALL; do [ -f results/$s/rangecode_v2a_r1_g6.json ] || $PY -u range_code2.py --scene $s --group 6 --gpu --threads 16 > $L/rc2_${s}.log 2>&1; echo "$(date -Is) rc2 $s $( [ -f results/$s/rangecode_v2a_r1_g6.json ] && echo ok || echo FAIL)" >> $G; done
run() { s=$1; cfg=$2; tau=$3; na=4; [ $cfg = T6R11 ] && na=2
  out=results/$s/cold3_v2a_r1_${cfg}_tau${tau}_r4.json
  [ -f $out ] || start_locked 6000 4 $L/cold3f_${s}_${cfg}_${tau}.log $PY -u eval_cold3.py --scene $s --cfg $cfg --tau $tau --n-anchor $na --n-ev 3 --n-mds 2 --phy phy --view-nbr --t-ent 1.73 --suffix _r4
  echo "$(date -Is) cold3f $cfg tau$tau $s $( [ -f $out ] && echo ok || echo FAIL)" >> $G
}
worker() { for s in "$@"; do for cfg in T8R8 T6R11; do for tau in 2.5 10; do run $s $cfg $tau; done; done; done; }
worker kitchen garden stump room counter bonsai &
sleep 200; worker truck train playroom bicycle drjohnson &
wait; echo "$(date -Is) cold3f done" >> $G
