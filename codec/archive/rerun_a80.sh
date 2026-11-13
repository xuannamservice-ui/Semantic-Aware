#!/bin/bash
# Garden, concealment-aware 80k anchors: start after the 150k run has left training (GPU memory), then evaluate.
cd "$(dirname "$0")"
export CUDA_HOME=/usr/local/cuda-12.6 PATH=/usr/local/cuda-12.6/bin:$PATH TORCH_CUDA_ARCH_LIST=8.9
PY=~/.venvs/q1_2nd/bin/python; L=results/logs; s=garden
until [ -f results/$s/metrics_layered_a150.json ]; do sleep 60; done
mv $L/layered_${s}_a80.log $L/layered_${s}_a80_oom.log
$PY -u train_layered.py --scene $s --tag _a80 --anchors 80000 --aware 1 > $L/layered_${s}_a80.log 2>&1 &&
for c in T6R11 T8R8; do $PY -u eval_layered.py --scene $s --tag _a80 --cfg $c --s2 2.5 --n-pat 6 > $L/evl_${s}_a80_$c.log 2>&1; done
echo "DONE $(date -Is)" > $L/GARDEN_A80_DONE
