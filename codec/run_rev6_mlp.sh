#!/bin/bash
# Revision 6, follow-up with the jointly trained MLP (_v2m_r1) on all 11 scenes:
#  B  stage latencies of Table tab:latency in one session incl. the MLP rows (bench_latency.py --mlp 1)
#     -> results/bench_latency_rev6.json
#  A  per-frame session latency with the MLP on every scene (bench_e2e_frames.py --nets mlp; T8R8 tau 10 / 2.5 ms and
#     T6R11 tau 10 ms with concealment, 3 repetitions), concurrent sessions N = 1, 2, 3
#     -> results/rev6_latency_session_mlp.json, results/rev6_latency_conc_mlp.json
#  C  one-frame column of the executed cold start repeated twice (margin 0, chunk 8192, suffixes _run2, _run3) and once
#     with a 0.5-ms safety margin (_m05), then the pooled runs -> results/rev6_cold4_r5_runs.json
# Every timing run waits for a GPU without other processes and holds /tmp/q1_latency.lock.
# usage: nohup ./run_rev6_mlp.sh [A|B|C ...] > results/logs/rev6_mlp.log 2>&1 &
cd "$HOME/NamNX'sPaper/Q1_2nd/codec"
export CUDA_HOME=/usr/local/cuda-12.6 PATH=/usr/local/cuda-12.6/bin:$PATH TORCH_CUDA_ARCH_LIST=8.9
PY=~/.venvs/q1_2nd/bin/python; R=results/rev6_raw; L=results/logs; mkdir -p $R $L
SC="garden bicycle stump room counter kitchen bonsai truck train playroom drjohnson"
STEPS="${*:-B A C}"
idle() { [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; }
locked() { until idle; do sleep 60; done; flock /tmp/q1_latency.lock "$@"; }
for step in $STEPS; do
 case $step in
 B)
  [ -f results/bench_latency_rev6.json ] || locked $PY -u bench_latency.py --out results/bench_latency_rev6.json --lost 5000,10000,20000,30000 \
      --A-vis 110000 --anchors 250000 --n-warm 50 --n-rep 2000 --compile 1 --mlp 1 > $L/rev6_bench_latency.log 2>&1
  echo "$(date -Is) B $( [ -f results/bench_latency_rev6.json ] && echo ok || echo FAIL)" ;;
 A)
  for s in $SC; do
    [ -f $R/mlp_session_$s.json ] || locked $PY -u bench_e2e_frames.py --mode session --scenes $s --nets mlp \
        --runs T8R8:10:conceal,T8R8:2.5:conceal,T6R11:10:conceal --reps 3 --chains 10 --chains-t6 3 --out-json $R/mlp_session_$s.json > $L/rev6_mlp_session_$s.log 2>&1
    echo "$(date -Is) A session $s $( [ -f $R/mlp_session_$s.json ] && echo ok || echo FAIL)"
  done
  conc() { N=$1; shift; B=/tmp/rev6_mlp_bar_$N; rm -rf $B
    $PY -c "import argparse; from bench_e2e_frames import quiet; print(quiet([], argparse.Namespace(no_probe=False), 'pre-mlpconc$N'))"
    for s in "$@"; do
      $PY -u bench_e2e_frames.py --mode session --scenes $s --nets mlp --runs T8R8:10:conceal --chains 3 --no-probe --barrier $B --n-procs $N \
        --duration 60 --out-json $R/mlpconc${N}_$s.json > $L/rev6_mlpconc${N}_$s.log 2>&1 &
    done; wait; }
  export -f conc; export PY R L
  [ -f $R/mlpconc1_garden.json ] || locked bash -c 'conc 1 garden'
  [ -f $R/mlpconc2_room.json ] || locked bash -c 'conc 2 garden room'
  [ -f $R/mlpconc3_bicycle.json ] || locked bash -c 'conc 3 garden room bicycle'
  $PY bench_e2e_frames.py --aggregate "$R/mlp_session_*[a-z].json" --out-json results/rev6_latency_session_mlp.json
  $PY bench_e2e_frames.py --aggregate "$R/mlpconc*_*[a-z].json" --out-json results/rev6_latency_conc_mlp.json
  echo "$(date -Is) A done" ;;
 C)
  for spec in "0 _run2" "0 _run3" "0.5 _m05"; do
    set -- $spec; M=$1; SFX=$2
    for s in $SC; do
      [ -f results/$s/cold4_v2m_r1_T6R11_tau10_r5$SFX.json ] || locked $PY -u eval_cold4.py --scene $s --margin $M --chunk 8192 --exec-frames 1 \
          --suffix $SFX > $L/cold4_r5${SFX}_$s.log 2>&1
      echo "$(date -Is) C $SFX $s $(ls results/$s/cold4_v2m_r1_*_r5$SFX.json 2>/dev/null | wc -l) files"
    done
  done
  $PY cold4_r5_runs.py > $L/cold4_r5_runs.log 2>&1
  echo "$(date -Is) C done" ;;
 esac
done
echo "$(date -Is) all done"
