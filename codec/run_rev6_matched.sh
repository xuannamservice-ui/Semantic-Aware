#!/bin/bash
# Revision 6, reconciliation: cold start timed with the semantics of eval_cold3.py (eval_cold_matched.py; the paper's
# column: _v2a_r1, --view-nbr --t-ent 1.73 --mlp-fallback; the jointly trained MLP rows: _v2m_r1 --net-timing mlp),
# the view-restricted neighbour search in sessions, and the full-view first frame of every frame path (bench_e2e_frames
# --max-frames 1, both networks, all four T8R8 runs). All under /tmp/q1_latency.lock.
# usage: nohup ./run_rev6_matched.sh > results/logs/rev6_matched.log 2>&1 &
cd "$HOME/NamNX'sPaper/Q1_2nd/codec"
export CUDA_HOME=/usr/local/cuda-12.6 PATH=/usr/local/cuda-12.6/bin:$PATH TORCH_CUDA_ARCH_LIST=8.9
PY=~/.venvs/q1_2nd/bin/python; L=results/logs; R=results/rev6_raw
lk() { flock /tmp/q1_latency.lock "$@"; }
SC="garden bicycle stump room counter kitchen bonsai truck train playroom drjohnson"
rm -f results/stump/cold_matched_v2a_r1_smoke.json
for s in $SC; do
  [ -f results/$s/cold_matched_v2a_r1.json ] || lk $PY -u eval_cold_matched.py --scene $s --runs T8R8:2.5,T8R8:10,T6R11:2.5 --mlp-fallback > $L/rev6_matched_$s.log 2>&1
  echo "$(date -Is) matched $s $( [ -f results/$s/cold_matched_v2a_r1.json ] && echo ok || echo FAIL)"
done
for s in garden room playroom; do
  [ -f results/$s/cold_matched_v2m_r1_mlpj.json ] || lk $PY -u eval_cold_matched.py --scene $s --tag _v2m_r1 --net-timing mlp --runs T8R8:2.5 --knn-runs none --suffix _mlpj > $L/rev6_matched_mlpj_$s.log 2>&1
  echo "$(date -Is) matched mlpj $s $( [ -f results/$s/cold_matched_v2m_r1_mlpj.json ] && echo ok || echo FAIL)"
done
for s in $SC; do
  [ -f results/$s/knn_session.json ] || lk $PY -u eval_cold_matched.py --scene $s --mode knn-session --chains 10 > $L/rev6_knnses_$s.log 2>&1
  echo "$(date -Is) knn-session $s $( [ -f results/$s/knn_session.json ] && echo ok || echo FAIL)"
done
$PY eval_cold_matched.py --aggregate > $L/rev6_matched_agg.log 2>&1; echo "$(date -Is) matched aggregate"
for s in $SC; do
  [ -f $R/frame0_$s.json ] || lk $PY -u bench_e2e_frames.py --mode session --scenes $s --nets deploy,attn --attn-all-runs --max-frames 1 --chains 10 --reps 3 --no-probe \
    --runs T8R8:10:conceal,T8R8:10:arq+conceal,T8R8:2.5:conceal,T8R8:2.5:arq+conceal --out-json $R/frame0_$s.json > $L/rev6_frame0_$s.log 2>&1
  echo "$(date -Is) frame0 $s $( [ -f $R/frame0_$s.json ] && echo ok || echo FAIL)"
done
echo "$(date -Is) rev6 matched done"; touch $R/MATCHED_DONE
