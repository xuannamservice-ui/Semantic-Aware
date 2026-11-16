#!/bin/bash
# Cold start with the anytime fallback (eval_cold.py --anytime): the concealment schemes are re-rendered on the same
# windows as run_cold.sh; the schemes without concealment are unchanged. usage: ./run_cold_any.sh <cfg> <n_anchor>
cd "$HOME/NamNX'sPaper/Q1_2nd/codec"
export CUDA_HOME=/usr/local/cuda-12.6 PATH=/usr/local/cuda-12.6/bin:$PATH TORCH_CUDA_ARCH_LIST=8.9
PY=~/.venvs/q1_2nd/bin/python; L=results/logs; G=$L/guard_review3.log
ALL="garden stump room counter bonsai truck train playroom bicycle kitchen drjohnson"
FIVE="garden stump room counter bonsai"
ok() { (( $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1) >= 6000 )) && (( $(awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo) >= 6 )); }
run() { cfg=$1; tau=$2; na=$3; nev=$4; shift 4
  for s in "$@"; do
    out=results/$s/cold_v2a_r1_any_${cfg}_tau${tau}.json
    for try in 1 2 3; do
      [ -f $out ] && break
      until ok; do sleep 30; done
      $PY -u eval_cold.py --scene $s --cfg $cfg --tau $tau --n-anchor $na --n-ev $nev --anytime --schemes conceal,rep2_conceal,arq_conceal --suffix _any > $L/coldany_${s}_${cfg}_${tau}.log 2>&1 && break
      sleep 60
    done
    echo "$(date -Is) coldany $cfg tau$tau $s $( [ -f $out ] && echo ok || echo FAIL)" >> $G
  done
}
run $1 2.5 $2 3 $ALL
run $1 10 $2 2 $FIVE
echo "$(date -Is) coldany $1 done" >> $G
