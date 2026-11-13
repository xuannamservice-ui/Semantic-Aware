#!/bin/bash
# Model B with per-mode failure patterns stored (for MDS / in-frame ARQ baselines): tau_c 2.5 and 10 ms,
# 20 + 20 chains of 2000 x 250 slots (1e7 slots per tau_c), seeds 9600+ (2.5 ms) and 9700+ (10 ms).
cd "$(dirname "$0")"; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1; PY=~/venv/bin/python; L=results/logs; mkdir -p results/patterns
n=0
for i in $(seq 0 19); do for tau in 2.5 10; do
  base=$([ $tau = 2.5 ] && echo 9600 || echo 9700)
  ($PY -u run_traces_corr.py --s2 2.5 --tau $tau --chains 2000 --slots 250 --seed $((base+i)) --rho-model kolmogorov --tag _pat$i > $L/pat_${tau}_$i.log 2>&1) &
  n=$((n+1)); (( n % 12 == 0 )) && wait
done; done; wait
mv results/traces_corr_2.5_*ms_pat*.npz results/traces_corr_2.5_*ms_pat*.json results/patterns/ 2>/dev/null
echo DONE $(date -Is) > $L/PATTERNS_DONE
