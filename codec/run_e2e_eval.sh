#!/bin/bash
# Codeword-level failed-slot rate (eval_e2e.py) on the PHY records of phy_sim/run_e2e.py, in order of priority; every GPU
# run holds the latency lock, so the latency benchmarks never share the GPU with it.
# usage: ./run_e2e_eval.sh <folder with rec_*.npz>
cd "$(dirname "$0")"
PY=~/.venvs/q1_2nd/bin/python
P=${1:-phy_e2e}
L=results/logs
C=stream,token_fill,token_rate,token_fill_stag
run() {   # <suffix> <n_sample> <variant patterns>
  flock /tmp/q1_latency.lock $PY -u eval_e2e.py --phy-e2e $P --combos $C --threads 1 --skip-done --suffix $1 --n-sample $2 \
    --variants "$3" --summary > $L/e2e$1.log 2>&1
  echo "$(date -Is) e2e$1 rc=$?" >> $L/e2e_chain.log
}
run _p1 1000000 "T6R11_paper_hd3.8e-3,T6R11_integ_hd3.8e-3"
run _p2 4000 "T6R11_integ_rin165_rs*,T6R11_integ_rin160_rs*,T6R11_integ_rs*,T6R11_integ_jit_hd3.8e-3"
run _p3 4000 "T8R8_paper_hd3.8e-3,T8R8_integ_hd3.8e-3"
run _p4 2000 "T6R11_paper435_*,T6R11_paper_rs*,T6R11_integ_rin16?_hd3.8e-3,T6R11_integ_jit_rs*,T8R8_paper_rs*,T8R8_paper435_*,T8R8_integ_rs*,T8R8_integ_rin*,T8R8_integ_jit_*"
$PY -u eval_e2e.py --no-run --summary --summary-suffixes _p1,_p2,_p3,_p4 > $L/e2e_summary.log 2>&1
echo "$(date -Is) summary rc=$?" >> $L/e2e_chain.log
