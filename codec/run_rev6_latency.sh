#!/bin/bash
# Revision 6, latency evidence (tasks A and B): per-frame end-to-end timing in session and cold-start mode on all scenes
# (bench_e2e_frames.py), concurrent sessions (N = 1, 2, 4), scene switch (bench_switch.py), then the pooled summaries
# results/rev6_latency_{session,cold,conc}.json and results/rev6_switch.json.
# Every timing run holds /tmp/q1_latency.lock (shared with the other GPU jobs) and /tmp/rev6_mine.lock (excludes the
# temporal-quality runs of run_rev6_temporal.sh); the scripts probe the host and wait while it is disturbed.
# usage: [WAITPID=<pid>] nohup ./run_rev6_latency.sh > results/logs/rev6_latency.log 2>&1 &
cd "$HOME/NamNX'sPaper/Q1_2nd/codec"
export CUDA_HOME=/usr/local/cuda-12.6 PATH=/usr/local/cuda-12.6/bin:$PATH TORCH_CUDA_ARCH_LIST=8.9
PY=~/.venvs/q1_2nd/bin/python; R=results/rev6_raw; L=results/logs; mkdir -p $R $L
SC="garden bicycle stump room counter kitchen bonsai truck train playroom drjohnson"
[ -n "$WAITPID" ] && while kill -0 $WAITPID 2>/dev/null; do sleep 30; done    # an earlier run still in progress
locked() { flock /tmp/rev6_mine.lock flock /tmp/q1_latency.lock "$@"; }
for s in $SC; do
  [ -f $R/session_$s.json ] || locked $PY -u bench_e2e_frames.py --mode session --scenes $s --nets deploy,attn --reps 3 --chains 10 --chains-t6 3 --chains-attn 5 --out-json $R/session_$s.json > $L/rev6_session_$s.log 2>&1
  echo "$(date -Is) session $s $( [ -f $R/session_$s.json ] && echo ok || echo FAIL)"
  [ -f $R/cold_$s.json ] || locked $PY -u bench_e2e_frames.py --mode cold --scenes $s --nets deploy,attn --reps 3 --out-json $R/cold_$s.json > $L/rev6_cold_$s.log 2>&1
  echo "$(date -Is) cold $s $( [ -f $R/cold_$s.json ] && echo ok || echo FAIL)"
done
[ -f results/rev6_switch.json ] || locked $PY -u bench_switch.py --reps 5 > $L/rev6_switch.log 2>&1
echo "$(date -Is) switch $( [ -f results/rev6_switch.json ] && echo ok || echo FAIL)"
conc() { N=$1; shift; B=/tmp/rev6_bar_$N; rm -rf $B
  $PY -c "import argparse; from bench_e2e_frames import quiet; print(quiet([], argparse.Namespace(no_probe=False), 'pre-conc$N'))"
  for s in "$@"; do
    $PY -u bench_e2e_frames.py --mode session --scenes $s --nets deploy --runs T8R8:10:conceal --chains 3 --no-probe --barrier $B --n-procs $N \
      --duration 60 --out-json $R/conc${N}_$s.json > $L/rev6_conc${N}_$s.log 2>&1 &
  done; wait; }
export -f conc; export PY R L
[ -f $R/conc1_garden.json ] || locked bash -c 'conc 1 garden'
[ -f $R/conc2_room.json ] || locked bash -c 'conc 2 garden room'
[ -f $R/conc4_truck.json ] || locked bash -c 'conc 4 garden room bicycle truck'
echo "$(date -Is) concurrency done"
$PY bench_e2e_frames.py --aggregate "$R/session_*[a-z].json" --out-json results/rev6_latency_session.json
$PY bench_e2e_frames.py --aggregate "$R/cold_*[a-z].json" --out-json results/rev6_latency_cold.json
$PY bench_e2e_frames.py --aggregate "$R/conc*_*[a-z].json" --out-json results/rev6_latency_conc.json
echo "$(date -Is) rev6 latency done"; touch $R/LATENCY_DONE
