#!/bin/bash
# Revision 6, temporal quality (task C): eval_temporal.py on all scenes (T8R8, tau_c = 10 and 2.5 ms, 10 chains), one
# scene at a time under /tmp/rev6_mine.lock (never together with this agent's timing runs), then the pooled summary
# results/rev6_temporal_T8R8.json; then the burst injection (eval_temporal.py --inject) on all scenes, without
# the lock (the timing runs probe the GPU and wait or retry while it is used). usage: nohup ./run_rev6_temporal.sh > results/logs/rev6_temporal.log 2>&1 &
cd "$HOME/NamNX'sPaper/Q1_2nd/codec"
export CUDA_HOME=/usr/local/cuda-12.6 PATH=/usr/local/cuda-12.6/bin:$PATH TORCH_CUDA_ARCH_LIST=8.9
PY=~/.venvs/q1_2nd/bin/python; L=results/logs
for s in garden bicycle stump room counter kitchen bonsai truck train playroom drjohnson; do
  [ -f results/$s/temporal_T8R8.json ] || flock /tmp/rev6_mine.lock $PY -u eval_temporal.py --scene $s --chains 10 > $L/rev6_temporal_$s.log 2>&1
  echo "$(date -Is) temporal $s $( [ -f results/$s/temporal_T8R8.json ] && echo ok || echo FAIL)"
done
for s in stump garden bicycle room counter kitchen bonsai truck train playroom drjohnson; do
  [ -f results/$s/temporal_inject_T8R8.json ] || $PY -u eval_temporal.py --scene $s --inject > $L/rev6_inject_$s.log 2>&1
  echo "$(date -Is) inject $s $( [ -f results/$s/temporal_inject_T8R8.json ] && echo ok || echo FAIL)"
done
$PY eval_temporal.py --aggregate --out-json results/rev6_temporal_T8R8.json
echo "$(date -Is) rev6 temporal done"
