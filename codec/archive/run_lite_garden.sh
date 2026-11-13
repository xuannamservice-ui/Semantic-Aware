#!/bin/bash
# Lightweight concealment network on the frozen concealment-aware garden representation, started when at least
# 12 GB of GPU memory are free (retries after an out-of-memory failure), then trace-driven evaluation.
cd "$(dirname "$0")"
export CUDA_HOME=/usr/local/cuda-12.6 PATH=/usr/local/cuda-12.6/bin:$PATH TORCH_CUDA_ARCH_LIST=8.9
PY=~/.venvs/q1_2nd/bin/python; L=results/logs
gpu_free() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | awk '{printf "%d", $1/1024}'; }
for try in 1 2 3 4 5 6; do
  until (( $(gpu_free) >= 12 )); do sleep 60; done
  $PY -u train_lite.py --scene garden --tag _a150 > $L/lite_garden_a150.log 2>&1 && break
  grep -q OutOfMemoryError $L/lite_garden_a150.log || exit 1
  mv $L/lite_garden_a150.log $L/lite_garden_a150_oom$try.log; sleep 300
done
$PY -u eval_layered.py --scene garden --tag _a150 --cfg T6R11 --s2 2.5 --n-pat 6 --suffix _lite > $L/evl_garden_a150_lite_T6R11.log 2>&1
echo "DONE $(date -Is)" > $L/LITE_GARDEN_DONE
