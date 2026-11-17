#!/bin/bash
# Cold start v3 (eval_cold3.py): per-view schedule, rendered MDS code, trained-model timing; T8R8 and T6R11 at
# tau_c = 2.5 ms on all scenes, at 10 ms on five scenes. One job at a time (host RAM). usage: ./run_cold3.sh
cd "$HOME/NamNX'sPaper/Q1_2nd/codec"
export CUDA_HOME=/usr/local/cuda-12.6 PATH=/usr/local/cuda-12.6/bin:$PATH TORCH_CUDA_ARCH_LIST=8.9
source ./gpu_lock.sh; PY=~/.venvs/q1_2nd/bin/python; L=results/logs; G=$L/guard_review3.log
ALL="garden stump room counter bonsai truck train playroom bicycle kitchen drjohnson"
FIVE="garden stump room counter bonsai"
f=results/garden/cold3_v2a_r1_T8R8_tau2.5.json      # remove the smoke test (n_ev = 1)
[ -f $f ] && python3 -c "import json,sys; sys.exit(0 if json.load(open(\"$f\"))[\"n_ev\"] == 1 else 1)" && mv $f ${f%.json}_smoke.json
run() { cfg=$1; tau=$2; na=$3; shift 3
  for s in "$@"; do
    out=results/$s/cold3_v2a_r1_${cfg}_tau${tau}.json
    [ -f $out ] || start_locked 5000 5 $L/cold3_${s}_${cfg}_${tau}.log $PY -u eval_cold3.py --scene $s --cfg $cfg --tau $tau --n-anchor $na --n-ev 3 --n-mds 2 --phy phy
    echo "$(date -Is) cold3 $cfg tau$tau $s $( [ -f $out ] && echo ok || echo FAIL)" >> $G
  done
}
run T8R8 2.5 4 $ALL
run T6R11 2.5 2 $ALL
run T8R8 10 4 $FIVE
run T6R11 10 2 $FIVE
echo "$(date -Is) cold3 done" >> $G
