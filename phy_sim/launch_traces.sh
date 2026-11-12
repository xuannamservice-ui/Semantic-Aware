#!/bin/bash
# Model B: temporally correlated traces (Kolmogorov temporal correlation), after launch_all.sh has finished.
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
L=results/logs
while [ ! -f $L/ALLDONE ]; do sleep 60; done
jobs_list=()
for i in $(seq 0 19); do jobs_list+=("2.5 $((9100+i)) _s$i"); done
for i in $(seq 0 9); do jobs_list+=("10 $((9200+i)) _s$i"); done
run() { tau=$1; seed=$2; tag=$3; name=traces_corr_2.5_${tau}ms$tag
  echo "START $(date -Is)" > $L/$name.log
  python3 -u run_traces_corr.py --s2 2.5 --tau $tau --chains 2000 --slots 250 --seed $seed --rho-model kolmogorov --tag $tag >> $L/$name.log 2>&1
  echo "EXIT $? $(date -Is)" >> $L/$name.log; }
n=0
for j in "${jobs_list[@]}"; do run $j & n=$((n+1)); if (( n % 8 == 0 )); then wait; fi; done
wait
echo TRACESDONE $(date -Is) > $L/TRACESDONE
