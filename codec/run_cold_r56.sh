#!/bin/bash
# Cold start with the corrected fallback (conceal only if time permits, otherwise render what arrived): r5 = first view
# within one frame period, r6 = within two (one startup frame). Three workers on an otherwise idle GPU.
cd "$HOME/NamNX'sPaper/Q1_2nd/codec"
export CUDA_HOME=/usr/local/cuda-12.6 PATH=/usr/local/cuda-12.6/bin:$PATH TORCH_CUDA_ARCH_LIST=8.9
PY=~/.venvs/q1_2nd/bin/python; L=results/logs; G=$L/guard_review3.log
run() { s=$1; cfg=$2; tau=$3; fr=$4; sfx=$5; na=4; [ $cfg = T6R11 ] && na=2
  out=results/$s/cold3_v2a_r1_${cfg}_tau${tau}${sfx}.json
  for try in 1 2; do
    [ -f $out ] && break
    $PY -u eval_cold3.py --scene $s --cfg $cfg --tau $tau --n-anchor $na --n-ev 3 --n-mds 2 --phy phy --view-nbr --t-ent 1.73 --mlp-fallback --frames $fr --suffix $sfx > $L/cold3${sfx}_${s}_${cfg}_${tau}.log 2>&1
  done
  echo "$(date -Is) cold3$sfx $cfg tau$tau $s $( [ -f $out ] && echo ok || echo FAIL)" >> $G
}
worker() { for s in "$@"; do for fr_sfx in "1 _r5" "2 _r6"; do set -- $fr_sfx; for cfg in T8R8 T6R11; do for tau in 2.5 10; do run $s $cfg $tau $fr_sfx; done; done; done; done; }
worker kitchen bonsai truck bicycle &
sleep 20; worker train counter playroom garden &
sleep 20; worker stump room drjohnson &
wait; echo "$(date -Is) cold r56 done" >> $G
