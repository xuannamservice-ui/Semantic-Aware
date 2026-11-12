#!/bin/bash
# Full PHY rerun of Access-2026-24041 on host stt (2026-11-12). Original seeds; new stages have their own seeds.
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
L=results/logs
r() { name=$1; shift; echo "START $(date -Is) $*" > $L/$name.log; python3 -u "$@" >> $L/$name.log 2>&1; echo "EXIT $? $(date -Is)" >> $L/$name.log; }
python3 -c "import numpy,scipy,sys;print(sys.version,numpy.__version__,scipy.__version__)" > $L/env.txt; uname -a >> $L/env.txt; lscpu | grep "Model name" >> $L/env.txt
# phase A
for s in 0.2 1.0 2.5; do r modeset_$s run_modeset.py --s2 $s --N 10000000 & done
r modeset_point_2.5 run_modeset.py --s2 2.5 --N 2000000 --point --tag point &
for s in 1.0 2.5; do r modeset_rin_$s run_modeset.py --s2 $s --N 2000000 --rin -155 --tag rin --suffix _rin & done
r smallball_2.5 run_smallball.py --s2 2.5 --N 4000000 --workers 8 --groups 8 &
r smallball_1.0 run_smallball.py --s2 1.0 --N 2000000 --workers 4 --groups 4 --seed 7702 &
wait
# phase B
r tolerance_2.5 run_tolerance.py --s2 2.5 --N 10000000 --workers 8 &
r tolerance_1.0 run_tolerance.py --s2 1.0 --N 4000000 --workers 4 --seed 8802 &
for st in oper; do for s in 0.2 1.0 2.5; do r ${st}_$s run_phy_results.py --stages $st --s2 $s & done; done
(r analytics_1.0 run_phy_results.py --stages analytics --s2 1.0; r jsweep_1.0 run_phy_results.py --stages jsweep --s2 1.0; r jsweep_2.5 run_phy_results.py --stages jsweep --s2 2.5) &
for s in 1.0 2.5; do (r sens_$s run_phy_results.py --stages sens --s2 $s; r power_$s run_phy_results.py --stages power --s2 $s) & done
for s in 1.0 2.5; do r alloc_T8R8_$s run_alloc.py --cfg T8R8 --s2 $s & done
wait
r summary plot_modeset.py
echo ALLDONE $(date -Is) > $L/ALLDONE
