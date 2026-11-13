#!/bin/bash
# EdgeConcealNet on the frozen concealment-aware garden representation (after the lite run), then evaluation.
cd "$(dirname "$0")"
export CUDA_HOME=/usr/local/cuda-12.6 PATH=/usr/local/cuda-12.6/bin:$PATH TORCH_CUDA_ARCH_LIST=8.9
PY=~/.venvs/q1_2nd/bin/python; L=results/logs
gpu_free() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | awk '{printf "%d", $1/1024}'; }
until [ -f $L/LITE_GARDEN_DONE ]; do sleep 60; done
for try in 1 2 3 4 5 6; do
  until (( $(gpu_free) >= 12 )); do sleep 60; done
  $PY -u train_lite.py --scene garden --tag _a150 --arch edge > $L/edge_garden_a150.log 2>&1 && break
  grep -q OutOfMemoryError $L/edge_garden_a150.log || exit 1
  mv $L/edge_garden_a150.log $L/edge_garden_a150_oom$try.log; sleep 300
done
$PY -u eval_layered.py --scene garden --tag _a150 --cfg T6R11 --s2 2.5 --n-pat 6 --suffix _edge > $L/evl_garden_a150_edge_T6R11.log 2>&1
echo "DONE $(date -Is)" > $L/EDGE_GARDEN_DONE
