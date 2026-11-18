#!/bin/bash
# Follow-up of run_rev6_latency.sh / run_rev6_rerun.sh: three concurrent sessions (between the measured 2 and 4) and the
# scene switch with 10 repetitions (the 5-repetition run is kept as results/rev6_switch_reps5.json); then the pooled
# concurrency summary. usage: nohup ./run_rev6_extra.sh > results/logs/rev6_extra.log 2>&1 &
cd "$HOME/NamNX'sPaper/Q1_2nd/codec"
export CUDA_HOME=/usr/local/cuda-12.6 PATH=/usr/local/cuda-12.6/bin:$PATH TORCH_CUDA_ARCH_LIST=8.9
PY=~/.venvs/q1_2nd/bin/python; R=results/rev6_raw; L=results/logs
until [ -f $R/RERUN_DONE ]; do sleep 30; done
locked() { flock /tmp/rev6_mine.lock flock /tmp/q1_latency.lock "$@"; }
conc() { N=$1; shift; B=/tmp/rev6_bar_$N; rm -rf $B
  $PY -c "import argparse; from bench_e2e_frames import quiet; print(quiet([], argparse.Namespace(no_probe=False), 'pre-conc$N'))"
  for s in "$@"; do
    $PY -u bench_e2e_frames.py --mode session --scenes $s --nets deploy --runs T8R8:10:conceal --chains 3 --no-probe --barrier $B --n-procs $N \
      --duration 60 --out-json $R/conc${N}_$s.json > $L/rev6_conc${N}_$s.log 2>&1 &
  done; wait; }
export -f conc; export PY R L
[ -f $R/conc3_bicycle.json ] || locked bash -c 'conc 3 garden room bicycle'
echo "$(date -Is) conc3 done"
[ -f results/rev6_switch_reps5.json ] || cp results/rev6_switch.json results/rev6_switch_reps5.json
locked $PY -u bench_switch.py --reps 10 > $L/rev6_switch10.log 2>&1
echo "$(date -Is) switch10 $( grep -q fresh_process results/rev6_switch.json && echo ok)"
$PY bench_e2e_frames.py --aggregate "$R/conc*_*[a-z].json" --out-json results/rev6_latency_conc.json
echo "$(date -Is) extra done"; touch $R/EXTRA_DONE
