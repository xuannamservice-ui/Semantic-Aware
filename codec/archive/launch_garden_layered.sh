#!/bin/bash
# Garden: concealment-aware layered model with 80k and 150k anchors, and the separate design (80k), then evaluation.
cd "$(dirname "$0")"
export CUDA_HOME=/usr/local/cuda-12.6 PATH=/usr/local/cuda-12.6/bin:$PATH TORCH_CUDA_ARCH_LIST=8.9
PY=~/.venvs/q1_2nd/bin/python; L=results/logs; s=garden
run() { tag=$1; shift
  $PY -u train_layered.py --scene $s --tag $tag "$@" > $L/layered_${s}$tag.log 2>&1 &&
  for c in T6R11 T8R8; do $PY -u eval_layered.py --scene $s --tag $tag --cfg $c --s2 2.5 --n-pat 6 > $L/evl_${s}${tag}_$c.log 2>&1; done; }
run _a80 --anchors 80000 --aware 1 &
run _a150 --anchors 150000 --aware 1 &
run _s80 --anchors 80000 --aware 0 &
wait
echo "DONE $(date -Is)" > $L/GARDEN_LAYERED_DONE
