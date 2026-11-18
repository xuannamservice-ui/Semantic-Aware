#!/bin/bash
# Revision 6, task D: three independent repetitions of the MLP rows of results/bench_latency_rev6.json (bench_latency.py
# --only-mlp 1, same harness, 50 warm-up + 2,000 timed runs, L = 5k/10k/20k/30k), one profiling run (launch-bound check),
# then the merge into results/bench_latency_rev6.json (new keys *_rep and mlp_rows_median; the old keys are kept).
# Starts after run_rev6_mlp.sh has finished task C. usage: nohup ./run_rev6_mlp_d.sh > results/logs/rev6_mlp_d.log 2>&1 &
cd "$HOME/NamNX'sPaper/Q1_2nd/codec"
export CUDA_HOME=/usr/local/cuda-12.6 PATH=/usr/local/cuda-12.6/bin:$PATH TORCH_CUDA_ARCH_LIST=8.9
PY=~/.venvs/q1_2nd/bin/python; L=results/logs
until grep -q "all done" $L/rev6_mlp.log; do sleep 60; done
idle() { [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; }
locked() { until idle; do sleep 60; done; flock /tmp/q1_latency.lock "$@"; }
ARGS="--lost 5000,10000,20000,30000 --A-vis 110000 --anchors 250000 --n-warm 50 --n-rep 2000 --compile 1 --only-mlp 1"
for r in 1 2 3; do
  [ -f results/bench_latency_rev6_mlprep$r.json ] || locked $PY -u bench_latency.py --out results/bench_latency_rev6_mlprep$r.json $ARGS > $L/rev6_mlprep$r.log 2>&1
  echo "$(date -Is) rep $r $(nvidia-smi --query-gpu=utilization.gpu,clocks.sm,temperature.gpu --format=csv,noheader) $(cat /proc/loadavg)"
done
[ -f results/bench_latency_rev6_mlpprof.json ] || locked $PY -u bench_latency.py --out results/bench_latency_rev6_mlpprof.json $ARGS --profile-mlp 1 > $L/rev6_mlpprof.log 2>&1
cp -n results/bench_latency_rev6.json results/bench_latency_rev6.json.single_run
$PY merge_mlp_reps.py results/bench_latency_rev6_mlprep{1,2,3}.json --profile results/bench_latency_rev6_mlpprof.json
echo "$(date -Is) D done"
