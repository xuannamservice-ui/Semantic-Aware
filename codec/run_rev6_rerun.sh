#!/bin/bash
# Re-times scenes whose session run overlapped with heavy CPU load of other jobs (run_rev6_latency.sh must have finished),
# keeps the earlier files as *.contended, and recomputes the pooled summaries.
# usage: WAITPID=<pid of run_rev6_latency.sh> nohup ./run_rev6_rerun.sh garden ... > results/logs/rev6_rerun.log 2>&1 &
cd "$HOME/NamNX'sPaper/Q1_2nd/codec"
export CUDA_HOME=/usr/local/cuda-12.6 PATH=/usr/local/cuda-12.6/bin:$PATH TORCH_CUDA_ARCH_LIST=8.9
PY=~/.venvs/q1_2nd/bin/python; R=results/rev6_raw; L=results/logs
[ -n "$WAITPID" ] && while kill -0 $WAITPID 2>/dev/null; do sleep 30; done
for s in "$@"; do
  [ -f $R/session_$s.json.contended ] || { mv $R/session_$s.json $R/session_$s.json.contended; mv $R/session_${s}_rows.json.gz $R/session_${s}_rows.json.gz.contended; }
  [ -f $R/session_$s.json ] || flock /tmp/rev6_mine.lock flock /tmp/q1_latency.lock $PY -u bench_e2e_frames.py --mode session --scenes $s --nets deploy,attn \
    --reps 3 --chains 10 --chains-t6 3 --chains-attn 5 --out-json $R/session_$s.json > $L/rev6_session_${s}_rerun.log 2>&1
  echo "$(date -Is) rerun session $s $( [ -f $R/session_$s.json ] && echo ok || echo FAIL)"
done
$PY bench_e2e_frames.py --aggregate "$R/session_*[a-z].json" --out-json results/rev6_latency_session.json
$PY bench_e2e_frames.py --aggregate "$R/cold_*[a-z].json" --out-json results/rev6_latency_cold.json
$PY bench_e2e_frames.py --aggregate "$R/conc*_*[a-z].json" --out-json results/rev6_latency_conc.json
echo "$(date -Is) rerun done"; touch $R/RERUN_DONE
