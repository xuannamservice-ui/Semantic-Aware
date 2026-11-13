#!/bin/bash
# Model B: temporally correlated traces (Kolmogorov temporal correlation), container sim-q1 on pve.
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=~/venv/bin/python
L=results/logs
$PY -c "import numpy,scipy,sys;print(sys.version,numpy.__version__,scipy.__version__)" > $L/env_traces.txt; lscpu | grep "Model name" >> $L/env_traces.txt
run() { tau=$1; seed=$2; tag=$3; name=traces_corr_2.5_${tau}ms$tag
  echo "START $(date -Is)" > $L/$name.log
  $PY -u run_traces_corr.py --s2 2.5 --tau $tau --chains 2000 --slots 250 --seed $seed --rho-model kolmogorov --tag $tag >> $L/$name.log 2>&1
  echo "EXIT $? $(date -Is)" >> $L/$name.log; }
jobs_list=()
for i in $(seq 0 19); do jobs_list+=("2.5 $((9100+i)) _s$i"); done
for i in $(seq 0 9); do jobs_list+=("10 $((9200+i)) _s$i"); done
n=0
for j in "${jobs_list[@]}"; do run $j & n=$((n+1)); if (( n % 10 == 0 )); then wait; fi; done
wait
echo TRACESDONE $(date -Is) > $L/TRACESDONE
