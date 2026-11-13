#!/bin/bash
# EdgeConcealNet trained on the frozen final (v2, joint) representation of the given scenes, then evaluation.
cd "$(dirname "$0")"
export CUDA_HOME=/usr/local/cuda-12.6 PATH=/usr/local/cuda-12.6/bin:$PATH TORCH_CUDA_ARCH_LIST=8.9
PY=~/.venvs/q1_2nd/bin/python; L=results/logs
mem_avail() { awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo; }
for s in "$@"; do
  until [ -f results/$s/eval_layered_v2a_r1_T8R8_2.5.json ]; do sleep 120; done
  until (( $(mem_avail) >= 12 )); do sleep 60; done
  [ -f results/$s/edge_v2a_r1.pt ] || $PY -u train_lite.py --scene $s --tag _v2a_r1 --arch edge > $L/edge_${s}_v2a_r1.log 2>&1 || continue
  for c in T6R11 T8R8; do $PY -u eval_layered.py --scene $s --tag _v2a_r1 --cfg $c --s2 2.5 --n-pat 6 --suffix _edge > $L/evl_${s}_v2a_r1_edge_$c.log 2>&1; done
done
