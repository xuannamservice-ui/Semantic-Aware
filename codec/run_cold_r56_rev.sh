#!/bin/bash
# Extra workers for run_cold_r56.sh in reverse order (last scenes, two-frame runs and tau_c = 10 ms first); a run is
# claimed with mkdir so that two extra workers never start the same run, and finished runs are skipped.
cd "$HOME/NamNX'sPaper/Q1_2nd/codec"
export CUDA_HOME=/usr/local/cuda-12.6 PATH=/usr/local/cuda-12.6/bin:$PATH TORCH_CUDA_ARCH_LIST=8.9
PY=~/.venvs/q1_2nd/bin/python; L=results/logs; G=$L/guard_review3.log; mkdir -p results/.claims
run() { s=$1; cfg=$2; tau=$3; fr=$4; sfx=$5; na=4; [ $cfg = T6R11 ] && na=2
  out=results/$s/cold3_v2a_r1_${cfg}_tau${tau}${sfx}.json
  [ -f $out ] && return
  mkdir results/.claims/${s}_${cfg}_${tau}${sfx} 2>/dev/null || return
  $PY -u eval_cold3.py --scene $s --cfg $cfg --tau $tau --n-anchor $na --n-ev 3 --n-mds 2 --phy phy --view-nbr --t-ent 1.73 --mlp-fallback --frames $fr --suffix $sfx > $L/cold3${sfx}_${s}_${cfg}_${tau}_rev.log 2>&1
  echo "$(date -Is) cold3$sfx $cfg tau$tau $s $( [ -f $out ] && echo ok || echo FAIL) (rev)" >> $G
}
worker() { for s in "$@"; do for fr_sfx in "2 _r6" "1 _r5"; do set -- $fr_sfx; for cfg in T6R11 T8R8; do for tau in 10 2.5; do run $s $cfg $tau $fr_sfx; done; done; done; done; }
worker bicycle truck bonsai kitchen &
sleep 15; worker garden playroom counter train &
sleep 15; worker drjohnson room stump &
wait; echo "$(date -Is) cold rev workers done" >> $G
