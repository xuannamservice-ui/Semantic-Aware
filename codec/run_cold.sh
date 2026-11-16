#!/bin/bash
# Cold start with real renders (eval_cold.py): T8R8 (four anchor copies) and T6R11 (two) at tau_c = 2.5 ms on all scenes
# and at 10 ms on five scenes. One process; waits for free GPU memory and RAM before every run.
cd "$HOME/NamNX'sPaper/Q1_2nd/codec"
export CUDA_HOME=/usr/local/cuda-12.6 PATH=/usr/local/cuda-12.6/bin:$PATH TORCH_CUDA_ARCH_LIST=8.9
PY=~/.venvs/q1_2nd/bin/python; L=results/logs; G=$L/guard_review3.log
ALL="garden stump room counter bonsai truck train playroom bicycle kitchen drjohnson"
FIVE="garden stump room counter bonsai"
ok() { (( $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1) >= 7000 )) && (( $(awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo) >= 6 )); }
run() { cfg=$1; tau=$2; na=$3; nev=$4; shift 4
  for s in "$@"; do
    out=results/$s/cold_v2a_r1_${cfg}_tau${tau}.json
    for try in 1 2 3; do
      [ -f $out ] && break
      until ok; do sleep 60; done
      $PY -u eval_cold.py --scene $s --cfg $cfg --tau $tau --n-anchor $na --n-ev $nev > $L/cold_${s}_${cfg}_${tau}.log 2>&1 && break
      sleep 120
    done
    echo "$(date -Is) cold $cfg tau$tau $s $( [ -f $out ] && echo ok || echo FAIL)" >> $G
  done
}
run T8R8 2.5 4 3 $ALL
run T6R11 2.5 2 3 $ALL
run T8R8 10 4 2 $FIVE
run T6R11 10 2 2 $FIVE
echo "$(date -Is) cold chain done" >> $G
