#!/bin/bash
# Deployment variant (EdgeConcealNet, cached encodings) on the frozen final representation of the given scenes, full
# precision only (--steps-ternary 0; garden and stump were trained with the same 10,000 full-precision steps), then the
# outage-slot evaluation on the T8R8 trace. One GPU job starts at a time (gpu_lock.sh). usage: ./run_edge_all.sh <scene> ...
cd "$HOME/NamNX'sPaper/Q1_2nd/codec"
export CUDA_HOME=/usr/local/cuda-12.6 PATH=/usr/local/cuda-12.6/bin:$PATH TORCH_CUDA_ARCH_LIST=8.9
source ./gpu_lock.sh; PY=~/.venvs/q1_2nd/bin/python; L=results/logs; G=$L/guard_review3.log
for s in "$@"; do
  [ -f results/$s/edge_v2a_r1.pt ] || start_locked 6000 4 $L/edge_${s}_v2a_r1.log $PY -u train_lite.py --scene $s --tag _v2a_r1 --arch edge --steps-ternary 0
  echo "$(date -Is) edge train $s $( [ -f results/$s/edge_v2a_r1.pt ] && echo ok || echo FAIL)" >> $G
  if [ -f results/$s/edge_v2a_r1.pt ] && [ ! -f results/$s/eval_layered_v2a_r1_edge_T8R8_2.5.json ]; then
    start_locked 6000 4 $L/evl_${s}_v2a_r1_edge_T8R8.log $PY -u eval_layered.py --scene $s --tag _v2a_r1 --cfg T8R8 --s2 2.5 --n-pat 6 --suffix _edge
  fi
  echo "$(date -Is) edge eval $s $( [ -f results/$s/eval_layered_v2a_r1_edge_T8R8_2.5.json ] && echo ok || echo FAIL)" >> $G
done
echo "$(date -Is) edge worker $1 done" >> $G
